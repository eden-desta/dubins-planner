import math
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple


TAU = 2 * math.pi # TAU = 2π, one full revolution in radians

@dataclass
class Pose:
    """Represents a 2D position and heading (in radians)."""
    x: float
    y: float
    heading: float  # radians

@dataclass
class Segment:
    kind: str      # 'L','R','S'
    param: float   # arc angle (rad, >=0) for L/R; length for S
    center: Optional[np.ndarray] = None
    start_theta: Optional[float] = None
    p0: Optional[np.ndarray] = None
    p1: Optional[np.ndarray] = None

@dataclass
class DubinsPath:
    path_type: str
    t: float  # arc length of the first segment
    p: float  # length of the middle segment (arc or straight)
    q: float  # arc length of the third segment
    length: float  # total length
    segments: List[Segment]  # should always have three segments

def plot_circle(center, radius, **kwargs):
    """
    Plot a circle representing a Dubins turning circle.
    Updated to accept standard matplotlib keyword arguments (alpha, linestyle, etc.).
    """
    # Create the circle points
    th = np.linspace(0, 2*math.pi, 100)
    pts = center + radius * np.array([np.cos(th), np.sin(th)]).T
    
    # Handle the old 'style' parameter if it's passed, mapping it to 'linestyle'
    if 'style' in kwargs:
        kwargs['linestyle'] = kwargs.pop('style')
    
    # Set a default color if none is provided
    if 'color' not in kwargs:
        kwargs['color'] = 'gray'
        
    # Plot using standard matplotlib args
    plt.plot(pts[:,0], pts[:,1], **kwargs)

def plot_arrow(p0, heading, length=0.5, color='black'):
    """
    Plot a small arrow to indicate a pose's heading direction.

    Parameters
    ----------
    p0 : array-like of shape (2,)
        Starting point of the arrow (x, y).
    heading : float
        Heading angle in radians.
    length : float
        Arrow length.
    color : str
        Arrow color.

    Notes
    -----
    - Computes a forward-pointing vector using (cos(heading), sin(heading)).
    - Draws an arrow from p0 → p1 with an arrowhead.
    """
    p1 = p0 + length * np.array([math.cos(heading), math.sin(heading)])
    plt.arrow(
        p0[0], p0[1],
        p1[0] - p0[0], p1[1] - p0[1],
        head_width=length * 0.075,
        length_includes_head=True,
        color=color
    )

def plot_line(p0, p1, color='black', style='-'):
    """
    Plot a straight line segment between points p0 and p1.

    Parameters
    ----------
    p0, p1 : array-like of shape (2,)
        Endpoints of the line segment.
    color : str
        Line color.
    style : str
        Line style.

    Notes
    -----
    - Useful for plotting internal/external tangents.
    - Also used for the straight 'S' part of CSC paths.
    """
    plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linestyle=style)

def setup_plot(title=""):
    """
    Create a clean square plotting area for Dubins diagrams.

    Parameters
    ----------
    title : str
        Title of the plot.

    Notes
    -----
    - Ensures x and y scale equally so circles remain circular.
    - Adds axes labels and a light grid for readability.
    - Creates a 6x6 inch figure to match geometric proportions.
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.grid(True, linestyle=':', alpha=0.6)

# --- Helper Math Functions ---
def ang(v: np.ndarray) -> float:
    return math.atan2(v[1], v[0])

def rot(v: np.ndarray, th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

def perp_ccw(v):
    """
    Return the 90° counter-clockwise (CCW) perpendicular of a 2D vector.

    Given a vector v = [x, y], the CCW perpendicular is:
        [-y, x]

    This is used when a leftward (CCW) perpendicular direction is required,
    such as identifying the correct side for tangent construction in
    left-turn paths like LSL or LSR.
    """
    return np.array([-v[1], v[0]])

def perp_cw(v):
    """
    Return the 90° clockwise (CW) perpendicular of a 2D vector.

    Given a vector v = [x, y], the CW perpendicular is:
        [y, -x]

    This is used when a rightward (CW) perpendicular direction is required,
    such as identifying the correct side for tangent construction in
    right-turn paths like RSR or RSL.
    """
    return np.array([v[1], -v[0]])

def mod2pi(theta):
    """Wrap angle to [0, 2π)."""
    return theta % (2 * math.pi)

def angle_on_circle(center, point):
    """
    Return the angle (0..2π) of a point around a circle center.
    """
    v = point - center
    return mod2pi(math.atan2(v[1], v[0]))

def arc_length(R, delta_theta):
    """Arc length for radius R and angle delta_theta (radians)."""
    return R * delta_theta

# --- Normalized Start Centers (Global Constants) ---
# In the normalized frame (Start at 0,0,0, R=1):
# The Left turn center is always at (0, 1)
CLs = np.array([0.0, 1.0])

# The Right turn center is always at (0, -1)
CRs = np.array([0.0, -1.0])

def normalize(start: Pose, goal: Pose, R: float) -> Tuple[np.ndarray, float]:
    """
    Transform the goal pose into the normalized frame relative to the start pose.

    In the normalized frame:
    1. The start position is at (0, 0).
    2. The start heading is 0.0 radians (facing +x).
    3. The turning radius is scaled to 1.0.

    Parameters
    ----------
    start : Pose
        The starting pose (x, y, heading).
    goal : Pose
        The goal pose (x, y, heading).
    R : float
        The minimum turning radius.

    Returns
    -------
    g_norm : np.ndarray
        The goal position (x, y) in the normalized frame.
    ga_norm : float
        The goal heading in the normalized frame (radians).
    """
    # 1. Translation: Shift so start is at (0,0)
    dx = goal.x - start.x
    dy = goal.y - start.y

    # 2. Rotation: Rotate by -start.heading to align start with +x axis
    # We use -heading because we are rotating the *world* into the *robot's* frame.
    # If the robot is facing 30 deg (pi/6), we rotate the world by -30 deg
    # so the robot ends up facing 0 deg.
    c = math.cos(-start.heading)
    s = math.sin(-start.heading)

    gx = c * dx - s * dy
    gy = s * dx + c * dy

    # 3. Scaling: Normalize distances by the turning radius R
    # This simplifies future math so we can assume R=1 everywhere.
    gx /= R
    gy /= R

    # 4. Heading: Normalize the relative heading
    # This is just the difference in angles.
    raw_heading_diff = goal.heading - start.heading
    ga_norm = raw_heading_diff % TAU  # Ensure it is in [0, 2pi)

    return np.array([gx, gy]), ga_norm

def goal_centers(g_norm: np.ndarray, ga_norm: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the centers of the Left and Right turning circles at the goal.

    Parameters
    ----------
    g_norm : np.ndarray
        The normalized goal position [x, y].
    ga_norm : float
        The normalized goal heading in radians.

    Returns
    -------
    CL_g : np.ndarray
        The center of the Left-turn circle at the goal.
    CR_g : np.ndarray
        The center of the Right-turn circle at the goal.
    """
    # Calculate the heading vector components
    # This represents the direction the robot is facing at the goal.
    c_ga = math.cos(ga_norm)
    s_ga = math.sin(ga_norm)

    # To find the LEFT circle center:
    # We move 1 radius (R=1) to the "left" of the heading.
    # The vector (cos, sin) rotated 90 deg CCW is (-sin, cos).
    CL_g = np.array([
        g_norm[0] - s_ga,  # x - sin(theta)
        g_norm[1] + c_ga   # y + cos(theta)
    ])

    # To find the RIGHT circle center:
    # We move 1 radius (R=1) to the "right" of the heading.
    # The vector (cos, sin) rotated 90 deg CW is (sin, -cos).
    CR_g = np.array([
        g_norm[0] + s_ga,  # x + sin(theta)
        g_norm[1] - c_ga   # y - cos(theta)
    ])

    return CL_g, CR_g

def external_csc(c1: np.ndarray, c2: np.ndarray, start_turn: str, goal_turn: str,
                 g: np.ndarray, ga: float, name: str) -> Optional[DubinsPath]:
    """
    General solver for LSL and RSR paths (Common External Tangents).
    """
    v = c2 - c1
    L = np.linalg.norm(v)
    if L < 1e-12:
        return None

    u = v / L
    p_len = L  # Straight length = distance between centers

    # --- 1. Start Turn (t) ---
    r0 = -c1  # Vector from c1 to Start(0,0)

    if start_turn == 'L':
        t1 = c1 + perp_cw(u)            # Tangent point on c1
        t_raw = ang(t1 - c1) - ang(r0)  # L-turn: End - Start (CCW)
    else:  # 'R'
        t1 = c1 + perp_ccw(u)           # Tangent point on c1
        t_raw = ang(r0) - ang(t1 - c1)  # R-turn: Start - End (CW)

    # FIXED: Use mod2pi for minimal positive arc length
    t_ang = mod2pi(t_raw)
    seg0 = Segment(start_turn, t_ang, center=c1, start_theta=ang(r0))

    # --- 2. Goal Turn (q) ---
    r_goal = g - c2  # Vector from c2 to Goal

    if goal_turn == 'L':
        t2 = c2 + perp_cw(u)            # Tangent point on c2
        q_raw = ang(r_goal) - ang(t2 - c2)
    else:  # 'R'
        t2 = c2 + perp_ccw(u)           # Tangent point on c2
        q_raw = ang(t2 - c2) - ang(r_goal)

    # FIXED: Use mod2pi for minimal positive arc length
    q_ang = mod2pi(q_raw)
    seg2 = Segment(goal_turn, q_ang, center=c2, start_theta=ang(t2 - c2))

    # --- 3. Assemble ---
    seg1 = Segment('S', p_len, p0=t1, p1=t2)
    total = t_ang + p_len + q_ang

    return DubinsPath(name, t_ang, p_len, q_ang, total, [seg0, seg1, seg2])

def internal_csc(
    c1: np.ndarray,
    c2: np.ndarray,
    start_turn: str,
    goal_turn: str,
    g: np.ndarray,
    ga: float,
    name: str
) -> Optional[DubinsPath]:
    v = c2 - c1
    L = np.linalg.norm(v)
    print("L =", L)

    # Need enough separation for internal tangent: L > 2R (R=1)
    if L < 2.0 - 1e-12:
        return None

    u = v / L
    beta = math.asin(min(1.0, 2.0 / L))  # sin(beta) = 2R / L, R=1
    best = None
    tol = 1e-6

    for sgn in (+1.0, -1.0):
        w = rot(u, sgn * beta)

        # ------------------------------------------------
        # 1) Tangent points t1, t2 from direction w
        # ------------------------------------------------
        if start_turn == 'L':
            t1 = c1 + perp_cw(w)
        else:  # 'R'
            t1 = c1 + perp_ccw(w)

        if goal_turn == 'L':
            t2 = c2 + perp_cw(w)
        else:  # 'R'
            t2 = c2 + perp_ccw(w)

        # Check they lie on the circles (radius ~ 1)
        r1 = t1 - c1
        r2 = t2 - c2
        if abs(np.linalg.norm(r1) - 1.0) > tol or abs(np.linalg.norm(r2) - 1.0) > tol:
            # Numerical safety, but with your math this should be fine
            continue

        # Check true tangency: line is perpendicular to radius at each point
        line = t2 - t1
        if abs(np.dot(line, r1)) > tol or abs(np.dot(line, r2)) > tol:
            # This is the non-tangent branch; skip it
            continue

        # ------------------------------------------------
        # 2) Start arc (t) – use proper mod2pi, no "shortest arc" hack
        # ------------------------------------------------
        r0 = -c1  # vector from c1 to start pose (start is at origin)

        theta_r0 = ang(r0)
        theta_t1 = ang(r1)

        if start_turn == 'L':
            # CCW: end - start
            t_ang = mod2pi(theta_t1 - theta_r0)
        else:  # 'R'
            # CW: start - end
            t_ang = mod2pi(theta_r0 - theta_t1)

        seg0 = Segment(start_turn, t_ang, center=c1, start_theta=theta_r0)

        # ------------------------------------------------
        # 3) Goal arc (q)
        # ------------------------------------------------
        r_goal = g - c2
        theta_goal = ang(r_goal)
        theta_t2 = ang(r2)

        if goal_turn == 'L':
            # CCW: end - start
            q_ang = mod2pi(theta_goal - theta_t2)
        else:  # 'R'
            # CW: start - end
            q_ang = mod2pi(theta_t2 - theta_goal)

        seg2 = Segment(goal_turn, q_ang, center=c2, start_theta=theta_t2)

        # ------------------------------------------------
        # 4) Straight segment (p) – must be positive
        # ------------------------------------------------
        p_len = float(np.dot(v, w))  # = L * cos(beta) = sqrt(L^2 - 4)

        if p_len < 1e-12:
            continue

        seg1 = Segment('S', p_len, p0=t1, p1=t2)

        # Debug so you can see what's happening
        print(
            f"Candidate {name} (sgn={sgn}): "
            f"t_ang={t_ang:.4f}, p_len={p_len:.4f}, q_ang={q_ang:.4f}"
        )

        total = t_ang + p_len + q_ang
        cand = DubinsPath(name, t_ang, p_len, q_ang, total, [seg0, seg1, seg2])

        if (best is None) or (cand.length < best.length):
            best = cand

    return best

def circle_intersections(c1, r1, c2, r2):
    """
    Compute the intersection points of two circles in 2D (if they exist).

    Circles:
        - Circle 1: center c1 (2D numpy array), radius r1
        - Circle 2: center c2 (2D numpy array), radius r2

    Returns:
        - []           : no intersection (separate, nested, or degenerate)
        - [p]          : exactly one intersection (tangent circles)
        - [p1, p2]     : two intersection points

    Geometry outline:
        1) Let v = c2 - c1 and d = ||v|| be the center-to-center vector and distance.
        2) Reject if:
           - d ≈ 0      (same center; infinite or no intersections)
           - d > r1+r2  (too far apart)
           - d < |r1-r2| (one circle completely inside the other without touching)
        3) Otherwise, the circles intersect in 1 or 2 points.
           - Along the line from c1 to c2, the chord of intersection has its midpoint
             at distance a from c1:
                 a = (r1^2 - r2^2 + d^2) / (2d)
           - The half-chord length (perpendicular offset) is:
                 h^2 = r1^2 - a^2
           - The midpoint p on the center line is:
                 p = c1 + a * (v / d)
           - A perpendicular unit vector n points from p to both intersection points.
             Using a CCW perpendicular: n = perp_ccw(v / d)
           - The intersection points are:
                 p + h * n,  p - h * n
    """
    # Vector between circle centers: from c1 to c2
    v = c2 - c1

    # Center distance (||v||)
    d = np.linalg.norm(v)

    # Quick rejection / degeneracy tests:
    # - d ~ 0: same center (degenerate)
    # - d > r1 + r2: centers too far apart (no intersection)
    # - d < |r1 - r2|: one circle lies completely inside the other
    if d < 1e-12 or d > r1 + r2 + 1e-12 or d < abs(r1 - r2) - 1e-12:
        # print("No intersection: d =", d, "r1 =", r1, "r2 =", r2)
        return []

    # Distance from c1 along the center line to the chord midpoint:
    # a = (r1^2 - r2^2 + d^2) / (2d)
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)

    # Squared half-chord length (perpendicular offset):
    # h^2 = r1^2 - a^2
    h2 = r1 * r1 - a * a

    # Numerical guard: clamp tiny negatives to zero
    if h2 < 0:
        h2 = 0.0
    h = math.sqrt(h2)

    # Unit vector from c1 to c2
    u = v / d

    # Base point p on the line of centers (midpoint of the chord)
    p = c1 + a * u

    # A unit normal to the center line (rotate u by +90°)
    n = perp_ccw(u)

    # If h == 0, the circles are tangent: exactly one intersection at p
    if h == 0.0:
        return [p]

    # Otherwise, two intersection points symmetric about the center line
    return [p + h * n, p - h * n]


def solve_ccc(c1: np.ndarray, c2: np.ndarray, start_turn: str, mid_turn: str, goal_turn: str,
              g: np.ndarray, ga: float, name: str) -> Optional[DubinsPath]:
    """
    General solver for LRL and RLR paths (Circle-Circle-Circle).
    """
    centers = circle_intersections(c1, 2.0, c2, 2.0)
    if not centers:
        return None

    best_path = None

    for cm in centers:
        # Tangent points are midpoints between centers
        t1 = 0.5 * (c1 + cm)
        t2 = 0.5 * (cm + c2)

        # --- Start Turn (t) ---
        r0 = -c1
        t_raw = ang(t1 - c1) - ang(r0) if start_turn == 'L' else ang(r0) - ang(t1 - c1)
        t_ang = mod2pi(t_raw)  # FIXED: Use mod2pi
        seg0 = Segment(start_turn, t_ang, center=c1, start_theta=ang(r0))

        # --- Middle Turn (p) ---
        v1 = t1 - cm
        v2 = t2 - cm
        p_raw = ang(v2) - ang(v1) if mid_turn == 'L' else ang(v1) - ang(v2)
        p_ang = mod2pi(p_raw)  # FIXED: Use mod2pi
        seg1 = Segment(mid_turn, p_ang, center=cm, start_theta=ang(v1))

        # --- Goal Turn (q) ---
        r_goal = g - c2
        q_raw = ang(r_goal) - ang(t2 - c2) if goal_turn == 'L' else ang(t2 - c2) - ang(r_goal)
        q_ang = mod2pi(q_raw)  # FIXED: Use mod2pi
        seg2 = Segment(goal_turn, q_ang, center=c2, start_theta=ang(t2 - c2))

        # Assemble
        total = t_ang + p_ang + q_ang
        cand = DubinsPath(name, t_ang, p_ang, q_ang, total, [seg0, seg1, seg2])

        if best_path is None or cand.length < best_path.length:
            best_path = cand

    return best_path

def build_LSL(g: np.ndarray, ga: float) -> Optional[DubinsPath]:
    """
    Build the Left-Straight-Left path.
    Uses Start Left Center (CLs) and Goal Left Center (CLg).
    """
    CLg, _ = goal_centers(g, ga)

    # Sequence: L -> L (External Tangent)
    return external_csc(CLs, CLg, 'L', 'L', g, ga, 'LSL')

def build_RSR(g: np.ndarray, ga: float) -> Optional[DubinsPath]:
    """
    Build the Right-Straight-Right path.
    Uses Start Right Center (CRs) and Goal Right Center (CRg).
    """
    _, CRg = goal_centers(g, ga)

    # Sequence: R -> R (External Tangent)
    return external_csc(CRs, CRg, 'R', 'R', g, ga, 'RSR')

def build_LSR(g: np.ndarray, ga: float) -> Optional[DubinsPath]:
    """
    Build the Left-Straight-Right path.
    Uses Start Left Center (CLs) and Goal Right Center (CRg).
    """
    _, CRg = goal_centers(g, ga)
    # Start=L, Goal=R
    return internal_csc(CLs, CRg, 'L', 'R', g, ga, 'LSR')

def build_RSL(g: np.ndarray, ga: float) -> Optional[DubinsPath]:
    """
    Build the Right-Straight-Left path.
    Uses Start Right Center (CRs) and Goal Left Center (CLg).
    """
    CLg, _ = goal_centers(g, ga)
    # Start=R, Goal=L
    return internal_csc(CRs, CLg, 'R', 'L', g, ga, 'RSL')

def build_LRL(g: np.ndarray, ga: float) -> Optional[DubinsPath]:
    """
    Build the Left-Right-Left path.
    Uses Start Left Center (CLs) and Goal Left Center (CLg).
    """
    CLg, _ = goal_centers(g, ga)
    # Sequence: L -> R -> L
    return solve_ccc(CLs, CLg, 'L', 'R', 'L', g, ga, 'LRL')

def build_RLR(g: np.ndarray, ga: float) -> Optional[DubinsPath]:
    """
    Build the Right-Left-Right path.
    Uses Start Right Center (CRs) and Goal Right Center (CRg).
    """
    _, CRg = goal_centers(g, ga)
    # Sequence: R -> L -> R
    return solve_ccc(CRs, CRg, 'R', 'L', 'R', g, ga, 'RLR')


def solve_all(g_norm: np.ndarray, ga_norm: float) -> Dict[str, Optional[DubinsPath]]:
    """
    Attempt to build all 6 path types for a given normalized goal.
    """
    return {
        'LSL': build_LSL(g_norm, ga_norm),
        'RSR': build_RSR(g_norm, ga_norm),
        'LSR': build_LSR(g_norm, ga_norm),
        'RSL': build_RSL(g_norm, ga_norm),
        'RLR': build_RLR(g_norm, ga_norm),
        'LRL': build_LRL(g_norm, ga_norm),
    }

def finalize_segments(sol: DubinsPath):
    """
    Populate the start_theta, p0, and p1 fields for each segment 
    so they can be easily plotted.
    """
    # Start at the normalized origin (0,0) with heading 0
    pos = np.array([0.0, 0.0])
    yaw = 0.0
    
    for seg in sol.segments:
        if seg.kind in ('L', 'R'):
            # Arc Segment
            seg.center = seg.center if seg.center is not None else np.zeros(2)
            seg.start_theta = ang(pos - seg.center)
            
            is_left = (seg.kind == 'L')
            # Calculate end angle
            theta_end = seg.start_theta + (seg.param if is_left else -seg.param)
            
            # Update position to end of arc
            pos = seg.center + np.array([math.cos(theta_end), math.sin(theta_end)])
            # Update heading (tangent to circle)
            yaw = theta_end + (math.pi/2 if is_left else -math.pi/2)
            
        else:
            # Straight Segment
            if seg.p0 is None: 
                seg.p0 = pos.copy()
            
            # Calculate end point p1 based on length (param) and current yaw
            if seg.p1 is None:
                seg.p1 = seg.p0 + seg.param * np.array([math.cos(yaw), math.sin(yaw)])
            
            pos = seg.p1.copy()
            # Yaw does not change on a straight line

def trace_end_pose(solution: DubinsPath) -> Tuple[np.ndarray, float]:
    """
    Trace the path to find the final normalized position and heading.
    """
    pos = np.array([0.0, 0.0])
    yaw = 0.0
    
    for seg in solution.segments:
        if seg.kind in ('L', 'R'):
            is_left = (seg.kind == 'L')
            theta_start = seg.start_theta
            theta_end = theta_start + (seg.param if is_left else -seg.param)
            
            pos = seg.center + np.array([math.cos(theta_end), math.sin(theta_end)])
            yaw = theta_end + (math.pi/2 if is_left else -math.pi/2)
        else:
            pos = seg.p1.copy()
            
    return pos, mod2pi(yaw)

def build_and_verify(start: Pose, goal: Pose, R: float) -> Tuple[str, Optional[DubinsPath], float, float]:
    """
    Find the shortest valid Dubins path among all 6 types.
    Returns: (best_type_name, best_path_obj, pos_error, yaw_error)
    """
    # 1. Normalize
    g_norm, ga_norm = normalize(start, goal, R)
    
    # 2. Solve all types
    candidates = solve_all(g_norm, ga_norm)
    
    best = None
    
    for name, sol in candidates.items():
        if sol is None:
            continue
            
        # 3. Finalize and Trace
        finalize_segments(sol)
        epos, eyaw = trace_end_pose(sol)
        
        plot_solution(start, goal, R, sol, f"{name}")

        
        # 4. Check Errors (in normalized frame)
        pos_err = float(np.linalg.norm(epos - g_norm))
        yaw_err = float(min(mod2pi(eyaw - ga_norm), mod2pi(ga_norm - eyaw)))
        
        # Filter out invalid paths (should be handled by logic, but good as a safety net)
        if pos_err > 1e-4 or yaw_err > 1e-4:
            # print(f"Skipping {name}: High error (p={pos_err:.1e}, a={yaw_err:.1e})")
            continue
                        
        # 5. Pick Shortest
        if (best is None) or (sol.length < best[1].length):
            best = (name, sol, pos_err, yaw_err)
            
    if best is None:
        return "No Path", None, 0.0, 0.0
        
    return best

def plot_solution(start: Pose, goal: Pose, R: float, sol: DubinsPath, title_prefix: str = ""):
    """
    Plot the full Dubins solution in the world frame with enhanced visualization.
    """
    # 1. Setup Plot
    total_len = sol.length * R
    full_title = f"{title_prefix} | R={R:.1f} | Total Dist={total_len:.2f}"
    setup_plot(full_title)
    
    # 2. Helper: Calculate World Frame Circle Centers
    def get_circles(p: Pose):
        # Heading vector (cos, sin)
        c, s = math.cos(p.heading), math.sin(p.heading)
        # Left center: (+R * normal_left) -> normal_left is (-sin, cos)
        cl = np.array([p.x - R*s, p.y + R*c])
        # Right center: (+R * normal_right) -> normal_right is (sin, -cos)
        cr = np.array([p.x + R*s, p.y - R*c])
        return cl, cr

    CLs, CRs = get_circles(start)
    CLg, CRg = get_circles(goal)
    
    # 3. Determine which circles were used based on path type
    # Path type is string like 'LSL', 'RSR', etc.
    # Seg 1 uses Start circles, Seg 3 uses Goal circles.
    start_turn = sol.path_type[0] # 'L' or 'R'
    goal_turn  = sol.path_type[2] # 'L' or 'R'
    
    # Define styles
    used_style   = {'color': 'black',   'linestyle': '--', 'alpha': 0.8, 'linewidth': 0.8}
    unused_style = {'color': 'dimgray', 'linestyle': ':',  'alpha': 0.5, 'linewidth': 0.5}
    
    # Plot Start Circles
    plot_circle(CLs, R, **(used_style if start_turn == 'L' else unused_style))
    plot_circle(CRs, R, **(used_style if start_turn == 'R' else unused_style))
    
    # Plot Goal Circles
    plot_circle(CLg, R, **(used_style if goal_turn == 'L' else unused_style))
    plot_circle(CRg, R, **(used_style if goal_turn == 'R' else unused_style))
    
    # 4. Helper to convert normalized points to world points for the path
    def to_world(p_norm):
        x_s, y_s = p_norm[0] * R, p_norm[1] * R
        c, s = math.cos(start.heading), math.sin(start.heading)
        return np.array([c*x_s - s*y_s + start.x, s*x_s + c*y_s + start.y])

    # 5. Plot Path Segments with Lengths in Legend
    colors = ['tab:green', 'tab:orange', 'tab:red']
    
    for i, seg in enumerate(sol.segments):
        seg_length_world = seg.param * R
        label_str = f"Seg {i+1} ({seg.kind}): {seg_length_world:.2f}"
        
        if seg.kind in ('L', 'R'):
            # Sample arc
            is_left = (seg.kind == 'L')
            ts = np.linspace(0, seg.param, 50)
            thetas = seg.start_theta + (ts if is_left else -ts)
            
            # Arc points (normalized) -> World
            arc_x = seg.center[0] + np.cos(thetas)
            arc_y = seg.center[1] + np.sin(thetas)
            pts = np.array([to_world(np.array([x, y])) for x, y in zip(arc_x, arc_y)])
            plt.plot(pts[:,0], pts[:,1], color=colors[i], linewidth=2.5, label=label_str)
            
        else:
            # Straight segment
            p0 = to_world(seg.p0)
            p1 = to_world(seg.p1)
            plot_line(p0, p1, color=colors[i], style='-')
            # Draw line again just for legend labeling (plot_line doesn't support label arg in all versions)
            plt.plot([p0[0], p1[0]], [p0[1], p1[1]], color=colors[i], linewidth=2.5, label=label_str)

    # 6. Plot Start/Goal Arrows & Annotations
    plot_arrow(np.array([start.x, start.y]), start.heading, length=R, color='black')
    plt.scatter(start.x, start.y, color='green', zorder=5)
    plt.text(start.x, start.y - R*0.4, "Start", fontweight='bold', ha='center')
    
    plot_arrow(np.array([goal.x, goal.y]), goal.heading, length=R, color='black')
    plt.scatter(goal.x, goal.y, color='red', zorder=5)
    plt.text(goal.x, goal.y - R*0.4, "Goal", fontweight='bold', ha='center')

    # Add text box for Start Pose details
    start_info = f"Start Pose:\nx={start.x:.2f}\ny={start.y:.2f}\nθ={math.degrees(start.heading):.1f}°"
    # Top-left corner
    plt.text(0.02, 0.98, start_info, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    end_info = f"End Pose:\nx={goal.x:.2f}\ny={goal.y:.2f}\nθ={math.degrees(goal.heading):.1f}°"
    # Top-right corner
    plt.text(0.98, 0.98, end_info, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.legend(loc='best', fontsize='small')
    

def main():
    R = 30.0
    start = Pose(0.0, 140.0, math.radians(150.0))
    goal  = Pose(50.0, 0.0, math.radians(33.0))

    # --- Run simulation ---
    name, sol, pe, ye = build_and_verify(start, goal, R)

    if sol is not None:
        plot_solution(start, goal, R, sol, f"Final Solution\n{name}")
    else:
        print(f"Could not find a valid Dubins path for this configuration.")

    plt.show()
    
if __name__=="__main__":
    main()