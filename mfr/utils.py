"""
Utility functions for OCC operations.
"""

import math
import numpy as np

from OCC.Core.gp import gp_Vec, gp_Dir, gp_Pnt, gp_XYZ
from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
    GeomAbs_Line,
    GeomAbs_Circle,
    GeomAbs_Ellipse,
    GeomAbs_Hyperbola,
    GeomAbs_Parabola,
    GeomAbs_BezierCurve,
    GeomAbs_BSplineCurve,
    GeomAbs_OffsetCurve,
    GeomAbs_OtherCurve,
)

# ============================================================================
# Vector conversion utilities
# ============================================================================

def gp_vec_to_numpy(vec):
    """Convert gp_Vec to numpy array."""
    return np.array([vec.X(), vec.Y(), vec.Z()])

def gp_dir_to_numpy(d):
    """Convert gp_Dir to numpy array."""
    return np.array([d.X(), d.Y(), d.Z()])

def gp_pnt_to_numpy(pnt):
    """Convert gp_Pnt to numpy array."""
    return np.array([pnt.X(), pnt.Y(), pnt.Z()])

def numpy_to_gp_vec(arr):
    """Convert numpy array to gp_Vec."""
    return gp_Vec(arr[0], arr[1], arr[2])

def numpy_to_gp_dir(arr):
    """Convert numpy array to gp_Dir."""
    return gp_Dir(arr[0], arr[1], arr[2])

def numpy_to_gp_pnt(arr):
    """Convert numpy array to gp_Pnt."""
    return gp_Pnt(arr[0], arr[1], arr[2])

# ============================================================================
# Vector math utilities
# ============================================================================

def angle_between_vectors(v1, v2):
    """Compute angle in radians between two vectors."""
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    if v1_norm < 1e-10 or v2_norm < 1e-10:
        return 0.0
    cos_theta = np.clip(np.dot(v1 / v1_norm, v2 / v2_norm), -1.0, 1.0)
    return math.acos(cos_theta)

def normalize_vector(v):
    """Normalize a vector."""
    v = np.asarray(v)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros_like(v)
    return v / norm

# ============================================================================
# Axis alignment utilities
# ============================================================================

def is_axis_aligned(normal, tolerance=0.1):
    """Check if normal is approximately aligned with a principal axis."""
    normal = normalize_vector(normal)
    abs_normal = np.abs(normal)
    max_component = np.max(abs_normal)
    # If the largest component is close to 1, it's axis-aligned
    return max_component >= (1.0 - tolerance)

def principal_axis_index(normal):
    """Return which principal axis (0=x, 1=y, 2=z) the direction is closest to."""
    normal = normalize_vector(normal)
    return np.argmax(np.abs(normal))

def principal_axis_name(normal):
    """Return the principal axis name ('x', 'y', 'z')."""
    idx = principal_axis_index(normal)
    return ['x', 'y', 'z'][idx]

def axis_alignment_scores(normal):
    """Return scores for alignment with x, y, z axes."""
    normal = normalize_vector(normal)
    return np.abs(normal)

def angle_to_principal_axis(normal):
    """Compute angle (in radians) to the nearest principal axis."""
    normal = normalize_vector(normal)
    abs_normal = np.abs(normal)
    max_idx = np.argmax(abs_normal)
    principal = np.zeros(3)
    principal[max_idx] = 1.0 if normal[max_idx] > 0 else -1.0
    return angle_between_vectors(normal, principal)

def is_45_degrees(normal, tolerance_deg=5):
    """Check if normal is approximately at 45 degrees to principal axes."""
    angle = angle_to_principal_axis(normal)
    target = math.pi / 4.0  # 45 degrees
    return abs(angle - target) < math.radians(tolerance_deg)

# ============================================================================
# Surface type mappings
# ============================================================================

SURFACE_TYPE_TO_NAME = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier_surface",
    GeomAbs_BSplineSurface: "bspline_surface",
    GeomAbs_SurfaceOfRevolution: "surface_of_revolution",
    GeomAbs_SurfaceOfExtrusion: "surface_of_extrusion",
    GeomAbs_OffsetSurface: "offset_surface",
    GeomAbs_OtherSurface: "other_surface",
}

NAME_TO_SURFACE_TYPE = {v: k for k, v in SURFACE_TYPE_TO_NAME.items()}

def surface_type_name(surface_type):
    """Convert GeomAbs_SurfaceType enum to string."""
    return SURFACE_TYPE_TO_NAME.get(surface_type, f"unknown({surface_type})")

def is_planar(surface_type):
    """Check if surface type is plane."""
    return surface_type == GeomAbs_Plane

def is_cylindrical(surface_type):
    """Check if surface type is cylinder."""
    return surface_type == GeomAbs_Cylinder

def is_conical(surface_type):
    """Check if surface type is cone."""
    return surface_type == GeomAbs_Cone

def is_spherical(surface_type):
    """Check if surface type is sphere."""
    return surface_type == GeomAbs_Sphere

def is_toroidal(surface_type):
    """Check if surface type is torus."""
    return surface_type == GeomAbs_Torus

# ============================================================================
# Curve type mappings
# ============================================================================

CURVE_TYPE_TO_NAME = {
    GeomAbs_Line: "line",
    GeomAbs_Circle: "circle",
    GeomAbs_Ellipse: "ellipse",
    GeomAbs_Hyperbola: "hyperbola",
    GeomAbs_Parabola: "parabola",
    GeomAbs_BezierCurve: "bezier_curve",
    GeomAbs_BSplineCurve: "bspline_curve",
    GeomAbs_OffsetCurve: "offset_curve",
    GeomAbs_OtherCurve: "other_curve",
}

def curve_type_name(curve_type):
    """Convert GeomAbs_CurveType enum to string."""
    return CURVE_TYPE_TO_NAME.get(curve_type, f"unknown({curve_type})")

def is_line(curve_type):
    """Check if curve type is line."""
    return curve_type == GeomAbs_Line

def is_circle(curve_type):
    """Check if curve type is circle."""
    return curve_type == GeomAbs_Circle

# ============================================================================
# Angle conversion
# ============================================================================

def rad_to_deg(rad):
    """Convert radians to degrees."""
    return math.degrees(rad)

def deg_to_rad(deg):
    """Convert degrees to radians."""
    return math.radians(deg)

# ============================================================================
# Comparison with tolerance
# ============================================================================

def is_approx_equal(a, b, tol=1e-6):
    """Check if two values are approximately equal."""
    return abs(a - b) < tol

def is_approx_zero(a, tol=1e-6):
    """Check if value is approximately zero."""
    return abs(a) < tol

def vectors_approx_equal(v1, v2, tol=1e-6):
    """Check if two vectors are approximately equal."""
    return np.allclose(v1, v2, atol=tol)

# ============================================================================
# Edge convexity/concavity classification
# ============================================================================

def classify_dihedral_angle(dihedral_rad, smooth_tol_deg=5):
    """Classify dihedral angle.

    Returns:
        str: 'convex', 'concave', or 'smooth'
    """
    smooth_tol = math.radians(smooth_tol_deg)
    # Dihedral angle is the angle between normals
    # Adjust for face orientation interpretation
    # For outward-pointing normals:
    #   dihedral ~ 0: convex (outside)
    #   dihedral ~ pi: concave (inside)
    # Actually need to think carefully:
    # The angle between two outward normals:
    #   - For a convex edge, faces point outward, angle between normals < 180
    #   - For a concave edge, the angle is > 180

    # Normalize to [0, pi]
    d = dihedral_rad % (2 * math.pi)
    if d > math.pi:
        d = 2 * math.pi - d

    if d < smooth_tol or d > (math.pi - smooth_tol):
        return 'smooth'
    elif d < math.pi / 2:
        return 'convex'
    else:
        return 'concave'

def is_convex_angle(dihedral_rad, smooth_tol_deg=5):
    """Check if dihedral angle indicates convex edge."""
    return classify_dihedral_angle(dihedral_rad, smooth_tol_deg) == 'convex'

def is_concave_angle(dihedral_rad, smooth_tol_deg=5):
    """Check if dihedral angle indicates concave edge."""
    return classify_dihedral_angle(dihedral_rad, smooth_tol_deg) == 'concave'

def is_smooth_angle(dihedral_rad, smooth_tol_deg=5):
    """Check if dihedral angle indicates smooth/tangent edge."""
    return classify_dihedral_angle(dihedral_rad, smooth_tol_deg) == 'smooth'
