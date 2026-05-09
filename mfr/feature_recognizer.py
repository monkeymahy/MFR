"""
Machining feature recognizer based on geometric kernel rules.

Detects three feature types from B-Rep topology:
  - Hole (通孔 / 盲孔 / 锥孔 / 沉头孔)
  - Boss  (圆柱凸台 / 棱柱凸台)
  - Chamfer (等距倒角 / 非等距倒角)

Usage:
    from mfr.feature_recognizer import recognize_features, FeatureType
    features = recognize_features(shape)       # TopoDS_Shape -> list[Feature]
"""

import math
import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Dict, Optional, Tuple, Set

from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE, TopAbs_IN
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopTools import (
    TopTools_IndexedMapOfShape,
    TopTools_IndexedDataMapOfShapeListOfShape,
)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Torus,
    GeomAbs_Sphere,
    GeomAbs_Line,
    GeomAbs_Circle,
)
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir

from .utils import (
    gp_pnt_to_numpy,
    gp_dir_to_numpy,
    gp_vec_to_numpy,
    angle_between_vectors,
    normalize_vector,
    surface_type_name,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOLERANCE = 1e-6
ANGLE_TOL = math.radians(5.0)   # 5 degree angular tolerance
FULL_CYL_TOL = 0.15             # tolerance for 2*pi U-range check
MATERIAL_SIDE_EPS = 0.01        # epsilon for material-side probe


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class FeatureType(IntEnum):
    HOLE = 0
    BOSS = 1
    CHAMFER = 2


FEATURE_TYPE_NAMES = {
    FeatureType.HOLE: "Hole",
    FeatureType.BOSS: "Boss",
    FeatureType.CHAMFER: "Chamfer",
}


class HoleSubType(IntEnum):
    THROUGH = 0      # 通孔
    BLIND = 1        # 盲孔
    TAPERED = 2      # 锥孔
    COUNTERBORE = 3  # 沉头孔


class BossSubType(IntEnum):
    CYLINDRICAL = 0  # 圆柱凸台
    PRISMATIC = 1    # 棱柱凸台


class ChamferSubType(IntEnum):
    SYMMETRIC = 0    # 等距倒角 (45 deg)
    ASYMMETRIC = 1   # 非等距倒角


@dataclass
class Feature:
    """A recognized machining feature."""
    feature_type: FeatureType
    subtype: int
    face_indices: List[int] = field(default_factory=list)
    axis: Optional[np.ndarray] = None       # cylinder/cone axis
    radius: float = 0.0                     # cylinder radius
    depth: float = 0.0                      # hole depth / boss height
    distance1: float = 0.0                  # chamfer distance d1
    distance2: float = 0.0                  # chamfer distance d2
    center: Optional[np.ndarray] = None     # approximate center point

    @property
    def type_name(self) -> str:
        return FEATURE_TYPE_NAMES.get(self.feature_type, "Unknown")

    @property
    def subtype_name(self) -> str:
        if self.feature_type == FeatureType.HOLE:
            return ["Through", "Blind", "Tapered", "Counterbore"][self.subtype]
        if self.feature_type == FeatureType.BOSS:
            return ["Cylindrical", "Prismatic"][self.subtype]
        if self.feature_type == FeatureType.CHAMFER:
            return ["Symmetric", "Asymmetric"][self.subtype]
        return "Unknown"

    def to_dict(self) -> dict:
        d = {
            "type": self.type_name,
            "subtype": self.subtype_name,
            "face_indices": self.face_indices,
        }
        if self.axis is not None:
            d["axis"] = self.axis.tolist()
        if self.center is not None:
            d["center"] = self.center.tolist()
        if self.radius > 0:
            d["radius"] = round(self.radius, 6)
        if self.depth > 0:
            d["depth"] = round(self.depth, 6)
        if self.distance1 > 0 or self.distance2 > 0:
            d["distance1"] = round(self.distance1, 6)
            d["distance2"] = round(self.distance2, 6)
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _face_area(face) -> float:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return props.Mass()


def _face_centroid(face) -> np.ndarray:
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    return gp_pnt_to_numpy(props.CentreOfMass())


def _face_normal_at_uv(face, u=None, v=None) -> np.ndarray:
    surf = BRepAdaptor_Surface(face)
    if u is None:
        u = (surf.FirstUParameter() + surf.LastUParameter()) / 2.0
    if v is None:
        v = (surf.FirstVParameter() + surf.LastVParameter()) / 2.0
    props = BRepLProp_SLProps(surf, u, v, 1, 1e-6)
    if props.IsNormalDefined():
        return gp_dir_to_numpy(props.Normal())
    # fallback from partial derivatives
    pnt = gp_Pnt()
    du = gp_Vec()
    dv = gp_Vec()
    surf.D1(u, v, pnt, du, dv)
    n = du.Crossed(dv)
    if n.SquareMagnitude() > 1e-10:
        n.Normalize()
    return gp_vec_to_numpy(n)


def _face_surface_type(face) -> int:
    return BRepAdaptor_Surface(face).GetType()


def _is_full_cylinder(face) -> bool:
    """Check if cylindrical face spans full 2*pi in U."""
    surf = BRepAdaptor_Surface(face)
    if surf.GetType() != GeomAbs_Cylinder:
        return False
    u_range = surf.LastUParameter() - surf.FirstUParameter()
    return abs(u_range - 2 * math.pi) < FULL_CYL_TOL


def _cylinder_params(face):
    """Return (radius, axis_dir, location) for a cylindrical face."""
    surf = BRepAdaptor_Surface(face)
    cyl = surf.Cylinder()
    radius = cyl.Radius()
    axis_dir = gp_dir_to_numpy(cyl.Axis().Direction())
    location = gp_pnt_to_numpy(cyl.Location())
    return radius, axis_dir, location


def _cone_params(face):
    """Return (half_angle, radius, axis_dir, location) for a conical face."""
    surf = BRepAdaptor_Surface(face)
    cone = surf.Cone()
    half_angle = cone.SemiAngle()
    ref_radius = cone.RefRadius()
    axis_dir = gp_dir_to_numpy(cone.Axis().Direction())
    location = gp_pnt_to_numpy(cone.Location())
    return half_angle, ref_radius, axis_dir, location


def _wire_count(face) -> int:
    """Count wires (loops) of a face."""
    count = 0
    ex = TopExp_Explorer(face, TopAbs_WIRE)
    while ex.More():
        count += 1
        ex.Next()
    return count


def _edge_count(face) -> int:
    count = 0
    ex = TopExp_Explorer(face, TopAbs_EDGE)
    while ex.More():
        count += 1
        ex.Next()
    return count


def _vectors_parallel(v1: np.ndarray, v2: np.ndarray, tol=0.05) -> bool:
    """Check if two unit vectors are approximately parallel (or anti-parallel)."""
    return abs(abs(np.dot(normalize_vector(v1), normalize_vector(v2))) - 1.0) < tol


def _vectors_same_direction(v1: np.ndarray, v2: np.ndarray, tol=0.05) -> bool:
    """Check if two unit vectors point approximately the same direction."""
    return np.dot(normalize_vector(v1), normalize_vector(v2)) > (1.0 - tol)


# ---------------------------------------------------------------------------
# Material-side detection
# ---------------------------------------------------------------------------

def _is_inner_surface(face, shape) -> bool:
    """Determine if a face is an inner surface (material on the inside).

    For a face with geometric normal n at point P:
      probe point = P - eps * n  (move inward along normal)
      If probe is inside the solid => material inside => inner surface (hole wall).

    We use BRepClass3d_SolidClassifier for point-in-solid test.
    """
    from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier

    normal = _face_normal_at_uv(face)
    centroid = _face_centroid(face)

    # Probe slightly inward along normal direction
    probe_point = centroid - MATERIAL_SIDE_EPS * normal
    gp_probe = gp_Pnt(probe_point[0], probe_point[1], probe_point[2])

    classifier = BRepClass3d_SolidClassifier(shape, gp_probe, TOLERANCE)
    is_inside = (classifier.State() == TopAbs_IN)

    return is_inside


# ---------------------------------------------------------------------------
# Build adjacency
# ---------------------------------------------------------------------------

@dataclass
class _EdgeAdj:
    """Edge adjacency between two faces."""
    face1: int
    face2: int
    dihedral: float        # radians [0, pi]
    edge_type: str         # "line" / "circle" / ...
    is_convex: bool = False
    is_concave: bool = False
    is_smooth: bool = False


def _build_face_list(shape) -> Tuple[List, List]:
    """Extract ordered face list and shape->index mapping.

    Returns:
        (faces, face_to_idx)  where face_to_idx is a list allowing
        lookup by _find_face_index using IsSame
    """
    face_map = TopTools_IndexedMapOfShape()
    topexp.MapShapes(shape, TopAbs_FACE, face_map)

    faces = []
    for i in range(1, face_map.Size() + 1):
        faces.append(face_map.FindKey(i))

    return faces, faces


def _find_face_index(target, faces, _face_to_idx_unused=None) -> Optional[int]:
    for idx, face in enumerate(faces):
        if target.IsSame(face):
            return idx
    return None


def _build_adjacency(shape, faces, face_to_idx) -> Dict[int, List[_EdgeAdj]]:
    """Build face adjacency map with dihedral angles."""
    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp.MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    adj: Dict[int, List[_EdgeAdj]] = {i: [] for i in range(len(faces))}

    for ei in range(1, edge_face_map.Size() + 1):
        face_list = edge_face_map.FindFromIndex(ei)
        if face_list.Size() != 2:
            continue

        it = iter(face_list)
        f1 = next(it)
        f2 = next(it)

        idx1 = _find_face_index(f1, faces, face_to_idx)
        idx2 = _find_face_index(f2, faces, face_to_idx)
        if idx1 is None or idx2 is None:
            continue

        # Compute dihedral angle
        n1 = _face_normal_at_uv(f1)
        n2 = _face_normal_at_uv(f2)
        dihedral = angle_between_vectors(n1, n2)

        # Edge type
        edge = edge_face_map.FindKey(ei)
        curve = BRepAdaptor_Curve(edge)
        etype = {GeomAbs_Line: "line", GeomAbs_Circle: "circle"}.get(curve.GetType(), "other")

        # Convexity classification
        smooth_tol = math.radians(5)
        is_smooth = dihedral < smooth_tol or dihedral > (math.pi - smooth_tol)
        is_convex = (not is_smooth) and dihedral < math.pi / 2
        is_concave = (not is_smooth) and dihedral >= math.pi / 2

        adj_info = _EdgeAdj(
            face1=idx1, face2=idx2, dihedral=dihedral,
            edge_type=etype, is_convex=is_convex,
            is_concave=is_concave, is_smooth=is_smooth,
        )
        adj[idx1].append(adj_info)
        adj[idx2].append(adj_info)

    return adj


# ---------------------------------------------------------------------------
# Hole recognition
# ---------------------------------------------------------------------------

def _recognize_holes(faces, adj, shape) -> List[Feature]:
    """Recognize hole features per the rule set in the prompt.

    Hole rule summary:
      - Core face: a full cylinder or cone face
      - Material side: inner surface (probe P - eps*n is inside solid)
      - Through hole: both boundary loops connect to outer surface, no bottom plane
      - Blind hole: one end has a planar bottom roughly perpendicular to axis
      - Tapered/Counterbore: consecutive coaxial cyl/cone faces with consistent material side
    """
    features: List[Feature] = []
    used: Set[int] = set()

    # --- Phase 1: Find all cylindrical/conical inner faces ---
    cyl_inner = []   # (idx, radius, axis, location)
    cone_inner = []  # (idx, half_angle, ref_radius, axis, location)

    for i, face in enumerate(faces):
        stype = _face_surface_type(face)

        if stype == GeomAbs_Cylinder and _is_full_cylinder(face):
            if _is_inner_surface(face, shape):
                r, axis, loc = _cylinder_params(face)
                cyl_inner.append((i, r, axis, loc))

        elif stype == GeomAbs_Cone:
            if _is_inner_surface(face, shape):
                ha, rr, axis, loc = _cone_params(face)
                cone_inner.append((i, ha, rr, axis, loc))

    # --- Phase 2: Group coaxial cylindrical + conical faces ---
    # Two faces are coaxial if their axes are parallel and share the same line
    all_inner = []
    for item in cyl_inner:
        all_inner.append(("cyl", item[0], item[1], 0.0, item[2], item[3]))
    for item in cone_inner:
        all_inner.append(("cone", item[0], item[1], item[2], item[3], item[4]))

    # Sort by axis direction for grouping
    coaxial_groups: List[List[int]] = []  # each group is list of indices into all_inner

    assigned = set()
    for ai, a in enumerate(all_inner):
        if ai in assigned:
            continue
        group = [ai]
        assigned.add(ai)
        a_kind, a_idx, a_r, a_ha, a_axis, a_loc = a

        for bi, b in enumerate(all_inner):
            if bi in assigned:
                continue
            b_kind, b_idx, b_r, b_ha, b_axis, b_loc = b

            if not _vectors_parallel(a_axis, b_axis):
                continue

            # Check coaxial: project location difference onto axis should be
            # along the axis, and perpendicular offset should be small
            diff = a_loc - b_loc
            axis_norm = normalize_vector(a_axis)
            along = np.dot(diff, axis_norm)
            perp = diff - along * axis_norm
            # For coaxial, perpendicular offset should be zero (or very small
            # relative to the radii).  Allow up to 1% of the larger radius.
            r_max = max(a_r, b_r, 0.01)
            if np.linalg.norm(perp) < r_max * 0.05:
                group.append(bi)
                assigned.add(bi)

        coaxial_groups.append(group)

    # --- Phase 3: Classify each coaxial group ---
    for group in coaxial_groups:
        group_indices = [all_inner[gi][1] for gi in group]  # face indices
        if any(idx in used for idx in group_indices):
            continue

        # Check if group has both cylinder and cone => tapered / counterbore
        has_cyl = any(all_inner[gi][0] == "cyl" for gi in group)
        has_cone = any(all_inner[gi][0] == "cone" for gi in group)

        # Gather all neighbor faces of the group
        group_set = set(group_indices)
        neighbor_faces: Set[int] = set()
        for fi in group_indices:
            for edge_adj in adj.get(fi, []):
                nb = edge_adj.face2 if edge_adj.face1 == fi else edge_adj.face1
                if nb not in group_set:
                    neighbor_faces.add(nb)

        # Determine through vs blind:
        #   Through: no planar neighbor perpendicular to axis
        #   Blind: at least one planar neighbor whose normal is parallel to axis
        #          (this is the bottom face)
        main_axis = all_inner[group[0]][4]  # axis of first face in group
        bottom_faces = []
        for nb in neighbor_faces:
            if _face_surface_type(faces[nb]) == GeomAbs_Plane:
                nb_normal = _face_normal_at_uv(faces[nb])
                if _vectors_parallel(nb_normal, main_axis, tol=0.15):
                    bottom_faces.append(nb)

        # Collect all face indices for this feature
        all_face_indices = list(group_indices)

        # Compute representative radius (largest cyl radius)
        cyl_radii = [all_inner[gi][2] for gi in group if all_inner[gi][0] == "cyl"]
        rep_radius = max(cyl_radii) if cyl_radii else 0.0

        # Compute depth for blind holes (distance along axis from top to bottom)
        depth = 0.0
        if bottom_faces:
            group_centroids = [_face_centroid(faces[fi]) for fi in group_indices]
            bottom_centroids = [_face_centroid(faces[nb]) for nb in bottom_faces]
            axis_dir = normalize_vector(main_axis)
            # Project centroids onto axis and take max span
            all_proj = [np.dot(c, axis_dir) for c in group_centroids + bottom_centroids]
            depth = max(all_proj) - min(all_proj)

        # Compute approximate center
        center = np.mean([_face_centroid(faces[fi]) for fi in group_indices], axis=0)

        if has_cone and has_cyl:
            # Tapered or counterbore hole
            # If cone half-angle is close to the angle between cyl and cone sections
            # => counterbore; otherwise => tapered
            cone_ha = [all_inner[gi][3] for gi in group if all_inner[gi][0] == "cone"]
            avg_cone_ha = np.mean(cone_ha) if cone_ha else 0.0
            if avg_cone_ha > math.radians(30):
                subtype = HoleSubType.COUNTERBORE
            else:
                subtype = HoleSubType.TAPERED
        elif bottom_faces:
            subtype = HoleSubType.BLIND
            all_face_indices.extend(bottom_faces)
        else:
            subtype = HoleSubType.THROUGH

        feat = Feature(
            feature_type=FeatureType.HOLE,
            subtype=subtype,
            face_indices=sorted(set(all_face_indices)),
            axis=normalize_vector(main_axis),
            radius=rep_radius,
            depth=depth,
            center=center,
        )
        features.append(feat)
        used.update(all_face_indices)
        used.update(bottom_faces)

    return features


# ---------------------------------------------------------------------------
# Boss recognition
# ---------------------------------------------------------------------------

def _recognize_bosses(faces, adj, shape) -> List[Feature]:
    """Recognize boss/pad features per the prompt rules.

    Boss rule summary:
      - Top face: planar, normal points in protrusion direction
      - Side faces: around the top face perimeter
        * Cylindrical boss: single full cylinder
        * Prismatic boss: multiple planar faces
      - Material side: all faces are outer surfaces (material inside)
      - Bottom loop: side faces' bottom boundary forms closed loop,
        connecting to the main body surface
      - Closed-loop condition: starting from any top edge, traverse
        adjacent side faces and return to start
    """
    features: List[Feature] = []
    used: Set[int] = set()

    # --- Phase 1: Find candidate top faces ---
    # Top face criteria: planar, outer surface (NOT inner), axis-aligned normal
    candidate_tops = []
    for i, face in enumerate(faces):
        if i in used:
            continue
        if _face_surface_type(face) != GeomAbs_Plane:
            continue
        if _is_inner_surface(face, shape):
            continue

        normal = _face_normal_at_uv(face)
        # Boss top face should have mostly concave edges connecting to sides
        concave_count = 0
        total_edges = 0
        for edge_adj in adj.get(i, []):
            nb = edge_adj.face2 if edge_adj.face1 == i else edge_adj.face1
            if nb in used:
                continue
            total_edges += 1
            if edge_adj.is_concave:
                concave_count += 1

        # A boss top face is surrounded by side faces via concave edges
        if total_edges >= 1 and concave_count >= 1:
            candidate_tops.append((i, normal, concave_count))

    # --- Phase 2: For each top face, collect side faces ---
    for top_idx, top_normal, _ in candidate_tops:
        if top_idx in used:
            continue

        # Walk concave edges from top face to find side faces
        side_faces = []
        for edge_adj in adj.get(top_idx, []):
            nb = edge_adj.face2 if edge_adj.face1 == top_idx else edge_adj.face1
            if nb in used:
                continue
            if edge_adj.is_concave:
                side_faces.append(nb)

        if not side_faces:
            continue

        # --- Phase 3: Verify closed-loop condition ---
        # Starting from top face edges, traverse side faces and check
        # if we can form a closed ring
        visited_sides = set()
        stack = list(side_faces)
        while stack:
            sf = stack.pop()
            if sf in visited_sides:
                continue
            visited_sides.add(sf)

            # Extend to neighbors via concave edges within the candidate set
            for edge_adj in adj.get(sf, []):
                nb = edge_adj.face2 if edge_adj.face1 == sf else edge_adj.face1
                if nb == top_idx or nb in visited_sides or nb in used:
                    continue
                # Side faces connect to each other via smooth/convex/concave edges
                # The key is they form a ring around the boss
                if edge_adj.is_smooth or edge_adj.is_convex:
                    # Check if this neighbor also touches the top face
                    # or another side face
                    if nb in side_faces or any(
                        ea.face2 == nb or ea.face1 == nb
                        for ea in adj.get(top_idx, [])
                    ):
                        stack.append(nb)

        side_faces = list(visited_sides)
        if not side_faces:
            continue

        # --- Phase 4: Classify boss type ---
        # Check if all side faces are a single full cylinder => cylindrical boss
        # Or multiple planar faces => prismatic boss
        has_cyl_side = False
        has_planar_side = False
        boss_axis = None
        boss_radius = 0.0

        for sf in side_faces:
            stype = _face_surface_type(faces[sf])
            if stype == GeomAbs_Cylinder and _is_full_cylinder(faces[sf]):
                has_cyl_side = True
                r, axis, loc = _cylinder_params(faces[sf])
                # Verify axis is roughly parallel to top face normal
                if _vectors_parallel(axis, top_normal, tol=0.15):
                    boss_axis = axis
                    boss_radius = r
            elif stype == GeomAbs_Plane:
                has_planar_side = True

        # Determine subtype
        if has_cyl_side and not has_planar_side:
            subtype = BossSubType.CYLINDRICAL
        elif has_planar_side:
            subtype = BossSubType.PRISMATIC
        elif has_cyl_side:
            subtype = BossSubType.CYLINDRICAL
        else:
            continue  # Not a recognizable boss pattern

        # Compute boss height (distance from top face centroid along normal)
        top_center = _face_centroid(faces[top_idx])
        # Find the lowest point of side faces along the protrusion direction
        protrusion_dir = normalize_vector(top_normal)
        top_proj = np.dot(top_center, protrusion_dir)

        bottom_proj = top_proj  # start with top as baseline
        for sf in side_faces:
            sf_center = _face_centroid(faces[sf])
            sf_proj = np.dot(sf_center, protrusion_dir)
            bottom_proj = min(bottom_proj, sf_proj)

        height = abs(top_proj - bottom_proj)

        # Center = top face centroid
        center = top_center

        all_indices = [top_idx] + side_faces
        feat = Feature(
            feature_type=FeatureType.BOSS,
            subtype=subtype,
            face_indices=sorted(set(all_indices)),
            axis=boss_axis if boss_axis is not None else protrusion_dir,
            radius=boss_radius,
            depth=height,
            center=center,
        )
        features.append(feat)
        used.update(all_indices)

    return features


# ---------------------------------------------------------------------------
# Chamfer recognition
# ---------------------------------------------------------------------------

def _recognize_chamfers(faces, adj, shape) -> List[Feature]:
    """Recognize chamfer features per the prompt rules.

    Chamfer rule summary:
      - Face type: a narrow planar strip, typically quadrilateral
      - Two long opposite edges E1, E2 shared with faces F1, F2
        (the two originally intersecting body faces)
      - Symmetric chamfer: angle(P_normal, F1_normal) == angle(P_normal, F2_normal)
        (typically 45 deg, dihedral 135 deg)
      - Asymmetric chamfer: distances d1, d2 differ but still a narrow strip
        replacing a sharp edge
      - Width (short dimension) is significantly smaller than the
        long edges
    """
    features: List[Feature] = []
    used: Set[int] = set()

    for i, face in enumerate(faces):
        if i in used:
            continue
        if _face_surface_type(face) != GeomAbs_Plane:
            continue

        area = _face_area(face)
        if area < TOLERANCE:
            continue

        # A chamfer face should have exactly 2 long edges and 2 short edges
        # More precisely: it should have exactly 4 edges, forming 2 pairs
        # of opposite edges
        edge_list = []
        ex = TopExp_Explorer(face, TopAbs_EDGE)
        while ex.More():
            edge = ex.Current()
            curve = BRepAdaptor_Curve(edge)
            u1 = curve.FirstParameter()
            u2 = curve.LastParameter()
            # Approximate edge length
            if curve.GetType() == GeomAbs_Line:
                p1 = curve.Value(u1)
                p2 = curve.Value(u2)
                length = np.linalg.norm(gp_pnt_to_numpy(p1) - gp_pnt_to_numpy(p2))
            elif curve.GetType() == GeomAbs_Circle:
                length = curve.Circle().Radius() * abs(u2 - u1)
            else:
                length = abs(u2 - u1) * 0.1  # rough
            edge_list.append((edge, length, curve.GetType()))
            ex.Next()

        num_edges = len(edge_list)
        if num_edges < 3:
            continue

        # Sort edges by length
        edge_list.sort(key=lambda x: x[1], reverse=True)

        # For a typical chamfer: 2 long edges + 2 short edges
        # The aspect ratio (longest / shortest) should be significant
        if num_edges == 4:
            longest = edge_list[0][1]
            shortest = edge_list[-1][1]
            if shortest < TOLERANCE:
                continue
            aspect = longest / shortest
            # Chamfers are narrow strips: aspect ratio should be > 2
            if aspect < 2.0:
                continue
        elif num_edges == 3:
            # Triangular chamfer is also possible but less common
            longest = edge_list[0][1]
            shortest = edge_list[-1][1]
            if shortest < TOLERANCE:
                continue
            aspect = longest / shortest
            if aspect < 2.0:
                continue
        else:
            # More edges - could still be a chamfer if narrow
            # Check if face area is small relative to perimeter
            perimeter = sum(e[1] for e in edge_list)
            if perimeter < TOLERANCE:
                continue
            compactness = 4 * math.pi * area / (perimeter ** 2)
            # Chamfer strips have low compactness
            if compactness > 0.3:
                continue

        # --- Check adjacency: chamfer face should connect two "parent" faces ---
        # The two longest edges should each be shared with a different face
        chamfer_normal = _face_normal_at_uv(face)

        # Find neighbors and their edge info
        neighbor_info = {}  # neighbor_idx -> (dihedral, edge_type, edge_length)
        for edge_adj in adj.get(i, []):
            nb = edge_adj.face2 if edge_adj.face1 == i else edge_adj.face1
            neighbor_info[nb] = edge_adj

        if len(neighbor_info) < 2:
            continue

        # The two "parent" faces F1, F2 are the ones connected via the long edges
        # We identify them as the two neighbors with the largest shared edge lengths
        # and non-smooth dihedral angles
        parent_candidates = []
        for nb, ea in neighbor_info.items():
            if ea.is_smooth:
                continue
            # Check: the neighbor face normal should make a significant angle
            # with the chamfer normal (not parallel)
            nb_normal = _face_normal_at_uv(faces[nb])
            angle_to_nb = angle_between_vectors(chamfer_normal, nb_normal)
            if angle_to_nb > math.radians(15):  # Not nearly coplanar
                parent_candidates.append((nb, ea, angle_to_nb))

        if len(parent_candidates) < 2:
            continue

        # Take the two parent candidates with the most significant connections
        parent_candidates.sort(key=lambda x: x[2], reverse=True)
        f1_idx, ea1, angle1 = parent_candidates[0]
        f2_idx, ea2, angle2 = parent_candidates[1]

        # --- Symmetric vs Asymmetric check ---
        # Symmetric: angle(chamfer_normal, F1_normal) == angle(chamfer_normal, F2_normal)
        # i.e., the chamfer plane bisects the dihedral angle of the two parent faces
        n_f1 = _face_normal_at_uv(faces[f1_idx])
        n_f2 = _face_normal_at_uv(faces[f2_idx])

        d1_angle = angle_between_vectors(chamfer_normal, n_f1)
        d2_angle = angle_between_vectors(chamfer_normal, n_f2)

        angle_diff = abs(d1_angle - d2_angle)

        # Compute distances d1, d2 (perpendicular distance from chamfer face
        # to each parent face, measured at centroid)
        centroid = _face_centroid(face)

        def _distance_to_plane(point, plane_face):
            """Perpendicular distance from a point to a planar face."""
            surf = BRepAdaptor_Surface(plane_face)
            if surf.GetType() != GeomAbs_Plane:
                return 0.0
            pln = surf.Plane()
            a, b, c, d = pln.Coefficients()
            return abs(a * point[0] + b * point[1] + c * point[2] + d)

        dist1 = _distance_to_plane(centroid, faces[f1_idx])
        dist2 = _distance_to_plane(centroid, faces[f2_idx])

        # Classify subtype
        if angle_diff < math.radians(10):
            subtype = ChamferSubType.SYMMETRIC
        else:
            subtype = ChamferSubType.ASYMMETRIC

        feat = Feature(
            feature_type=FeatureType.CHAMFER,
            subtype=subtype,
            face_indices=[i],
            distance1=dist1,
            distance2=dist2,
            center=centroid,
        )
        features.append(feat)
        used.add(i)

    return features


# ---------------------------------------------------------------------------
# Main recognition pipeline
# ---------------------------------------------------------------------------

def recognize_features(shape) -> Tuple[List[Feature], int]:
    """Recognize Hole, Boss, and Chamfer features from a B-Rep solid.

    Args:
        shape: TopoDS_Shape (should be a solid or compound of solids)

    Returns:
        (features, num_faces) tuple
    """
    faces, face_to_idx = _build_face_list(shape)
    adj = _build_adjacency(shape, faces, face_to_idx)

    features = []

    # Recognize holes first (they have the most distinctive geometry)
    features.extend(_recognize_holes(faces, adj, shape))

    # Recognize bosses
    features.extend(_recognize_bosses(faces, adj, shape))

    # Recognize chamfers
    features.extend(_recognize_chamfers(faces, adj, shape))

    return features, len(faces)


def recognize_features_from_step(step_path: str) -> Tuple[List[Feature], int]:
    """Convenience: load a STEP file and recognize features.

    Args:
        step_path: Path to .step / .stp file

    Returns:
        (features, num_faces) tuple
    """
    from OCC.Core.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:  # IFSelect_RetDone = 1
        raise RuntimeError(f"Failed to read STEP file: {step_path} (status={status})")

    reader.TransferRoots()
    shape = reader.OneShape()

    return recognize_features(shape)


def features_to_label(features: List[Feature], num_faces: int) -> List[int]:
    """Convert recognized features to a per-face label list.

    Each face gets a label:
        0 = none/background
        1 = Hole
        2 = Boss
        3 = Chamfer

    Args:
        features: List of Feature objects
        num_faces: Total number of faces in the shape

    Returns:
        List of length num_faces, each element in {0, 1, 2, 3}
    """
    labels = [0] * num_faces
    for feat in features:
        code = int(feat.feature_type) + 1  # HOLE=1, BOSS=2, CHAMFER=3
        for fi in feat.face_indices:
            if 0 <= fi < num_faces:
                labels[fi] = code
    return labels
