//! Two-dimensional nodal Discontinuous Galerkin (P1) on triangular meshes for
//! scalar conservation laws.
//!
//! # Equations
//!
//! * Linear advection `u_t + ∇·(β u) = 0` with constant velocity `β = (βx, βy)`.
//! * Inviscid Burgers `u_t + ∂_x(u²/2) = 0` (1D flux embedded along x), used for
//!   the Riemann / shock-speed test.
//!
//! # Discretisation
//!
//! Each triangle carries a **P1** field with three element-local DOFs stored as
//! the nodal values at the triangle vertices. The basis functions are the
//! barycentric coordinates `λ_0, λ_1, λ_2`; their gradients are constant on the
//! element. The semi-discrete scheme is
//!
//! ```text
//! M_T du/dt = V_T(u) − S_T(u),
//! ```
//!
//! where
//!
//! * `M_T = (|T|/12)·[[2,1,1],[1,2,1],[1,1,2]]` is the P1 mass matrix,
//! * `V_T(u)_i = ∫_T (f(u)·∇λ_i)`  is the volume term (`f` the physical flux),
//! * `S_T(u)_i = ∮_{∂T} (f̂·n) λ_i ds`  is the surface term with the numerical
//!   normal flux `f̂·n` (upwind for advection, Lax-Friedrichs for Burgers),
//!   integrated by 2-point Gauss-Legendre per edge (exact for the P1×P1 product).
//!
//! Time integration uses the explicit **SSP-RK3** scheme of Shu & Osher (1988),
//! whose convex-combination structure preserves the discrete maximum principle
//! when paired with the slope limiter in [`crate::dg::limiter_2d`].
//!
//! # Boundary conditions
//!
//! * [`DgBoundary::Periodic`] matches opposite boundary edges of an
//!   axis-aligned rectangle by translating edge midpoints (mass is conserved to
//!   round-off).
//! * [`DgBoundary::Compact`] treats the exterior trace on inflow boundary edges
//!   as a fixed far-field value (default `0.0`), suitable for compactly-supported
//!   data that never reaches the boundary.
//!
//! Reference: Cockburn & Shu, *The Runge-Kutta Discontinuous Galerkin Method*
//! (J. Comput. Phys. 1998); Hesthaven & Warburton, *Nodal DG Methods* (2008).

use crate::error::{PdeError, PdeResult};
use crate::mesh::TriMesh2d;

/// Scalar flux model for the 2D DG solver.
#[derive(Debug, Clone, Copy)]
pub enum DgFlux {
    /// Linear advection with constant velocity `(βx, βy)`.
    Advection { bx: f64, by: f64 },
    /// Inviscid Burgers with flux `f(u) = (u²/2, 0)` (1D along x).
    Burgers,
}

/// Boundary treatment for the DG solver.
#[derive(Debug, Clone, Copy)]
pub enum DgBoundary {
    /// Periodic matching of opposite boundary edges of `[x0,x1]×[y0,y1]`.
    Periodic {
        /// Domain extents used to translate matching edge midpoints.
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
    },
    /// Compact support: exterior trace on inflow edges equals `far_field`.
    Compact {
        /// Far-field exterior value (commonly `0.0`).
        far_field: f64,
    },
}

/// Pre-computed per-element geometry for the DG solver.
#[derive(Debug, Clone)]
struct ElemGeom {
    /// Vertex coordinates `[[x0,y0],[x1,y1],[x2,y2]]`.
    p: [[f64; 2]; 3],
    /// Triangle area (positive).
    area: f64,
    /// Constant gradients of the barycentric basis `∇λ_i` (`[[gx0,gy0],...]`).
    grad: [[f64; 2]; 3],
}

/// Topology of an interior/boundary edge for flux coupling.
#[derive(Debug, Clone)]
struct EdgeLink {
    /// Owner element index.
    elem: usize,
    /// Local edge index (0..3) in the owner; the edge opposite vertex `local_edge`.
    local_edge: usize,
    /// Local owner vertex indices forming the edge, in owner traversal order.
    own_lv: [usize; 2],
    /// Neighbour element index (`None` for an unmatched boundary edge).
    neigh: Option<usize>,
    /// Neighbour local vertex indices matching `own_lv` endpoints (same physical
    /// points), so the neighbour trace can be interpolated consistently.
    neigh_lv: [usize; 2],
    /// Neighbour centroid expressed in this element's frame (periodic shift
    /// already applied); `None` for unmatched boundary edges. Used by the
    /// geometry-aware slope limiter.
    neigh_centroid: Option<[f64; 2]>,
}

/// Per-element limiter stencil: for each of the three local edges, the
/// neighbour element index and its centroid expressed in the owner's frame
/// (periodic shift already applied), or `None` for an unmatched boundary edge.
pub type LimiterStencil = [Option<(usize, [f64; 2])>; 3];

/// A per-stage limiter hook: a function mutating the nodal field in place for a
/// given DG space (used to plug the slope/MPP limiters into the RK integrator).
type LimiterHook<'a> = &'a dyn Fn(&Dg2dSpace, &mut [f64]) -> PdeResult<()>;

/// 2D nodal DG space (P1) over a triangular mesh.
#[derive(Debug, Clone)]
pub struct Dg2dSpace {
    /// Number of triangles.
    pub n_elem: usize,
    geom: Vec<ElemGeom>,
    edges: Vec<EdgeLink>,
}

/// Local-edge `l` (opposite vertex `l`) connects local vertices `(l+1, l+2)` mod 3.
fn local_edge_vertices(l: usize) -> (usize, usize) {
    ((l + 1) % 3, (l + 2) % 3)
}

impl Dg2dSpace {
    /// Build the DG space (geometry + edge connectivity) for `mesh` under `bc`.
    pub fn new(mesh: &TriMesh2d, bc: DgBoundary) -> PdeResult<Self> {
        let n_elem = mesh.n_tri();
        if n_elem == 0 {
            return Err(PdeError::EmptyMesh("dg_2d: no triangles".into()));
        }
        let mut geom = Vec::with_capacity(n_elem);
        for e in 0..n_elem {
            geom.push(Self::element_geometry(mesh, e)?);
        }
        let edges = Self::build_edges(mesh, &geom, bc)?;
        Ok(Self {
            n_elem,
            geom,
            edges,
        })
    }

    fn element_geometry(mesh: &TriMesh2d, e: usize) -> PdeResult<ElemGeom> {
        let (v0, v1, v2) = mesh.tri(e)?;
        let (x0, y0) = mesh.node(v0)?;
        let (x1, y1) = mesh.node(v1)?;
        let (x2, y2) = mesh.node(v2)?;
        let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
        if area.abs() < 1.0e-14 {
            return Err(PdeError::SingularMatrix(format!(
                "dg_2d: degenerate triangle {e}, area={area}"
            )));
        }
        // Barycentric gradients: ∇λ_i = (b_i, c_i)/(2A) with
        // b_i = y_{i+1}-y_{i+2}, c_i = x_{i+2}-x_{i+1} (cyclic), A signed area.
        let xs = [x0, x1, x2];
        let ys = [y0, y1, y2];
        let two_a = 2.0 * area;
        let mut grad = [[0.0_f64; 2]; 3];
        for (i, gi) in grad.iter_mut().enumerate() {
            let ip1 = (i + 1) % 3;
            let ip2 = (i + 2) % 3;
            let b = ys[ip1] - ys[ip2];
            let c = xs[ip2] - xs[ip1];
            *gi = [b / two_a, c / two_a];
        }
        Ok(ElemGeom {
            p: [[x0, y0], [x1, y1], [x2, y2]],
            area: area.abs(),
            grad,
        })
    }

    fn build_edges(
        mesh: &TriMesh2d,
        geom: &[ElemGeom],
        bc: DgBoundary,
    ) -> PdeResult<Vec<EdgeLink>> {
        use std::collections::HashMap;
        let n_elem = mesh.n_tri();
        // Map canonical (min,max) global vertex pair → list of (elem, local_edge).
        let mut map: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
        let mut tri_verts = Vec::with_capacity(n_elem);
        for e in 0..n_elem {
            let (v0, v1, v2) = mesh.tri(e)?;
            tri_verts.push([v0, v1, v2]);
            for l in 0..3 {
                let (a, b) = local_edge_vertices(l);
                let (va, vb) = ([v0, v1, v2][a], [v0, v1, v2][b]);
                let key = if va < vb { (va, vb) } else { (vb, va) };
                map.entry(key).or_default().push((e, l));
            }
        }

        let mut edges: Vec<EdgeLink> = Vec::new();
        // Track which (elem,local) have been consumed by an interior pairing.
        let mut consumed: HashMap<(usize, usize), bool> = HashMap::new();

        for incident in map.values() {
            if incident.len() == 2 {
                let (e0, l0) = incident[0];
                let (e1, l1) = incident[1];
                consumed.insert((e0, l0), true);
                consumed.insert((e1, l1), true);
                // own endpoints (e0 owner) and matching neighbour locals.
                let c0 = centroid_of(&geom[e0]);
                let c1 = centroid_of(&geom[e1]);
                let (a0, b0) = local_edge_vertices(l0);
                let link0 = EdgeLink {
                    elem: e0,
                    local_edge: l0,
                    own_lv: [a0, b0],
                    neigh: Some(e1),
                    neigh_lv: Self::match_neigh(&tri_verts[e0], a0, b0, &tri_verts[e1]),
                    neigh_centroid: Some(c1),
                };
                edges.push(link0);
                let (a1, b1) = local_edge_vertices(l1);
                let link1 = EdgeLink {
                    elem: e1,
                    local_edge: l1,
                    own_lv: [a1, b1],
                    neigh: Some(e0),
                    neigh_lv: Self::match_neigh(&tri_verts[e1], a1, b1, &tri_verts[e0]),
                    neigh_centroid: Some(c0),
                };
                edges.push(link1);
            }
        }

        // Boundary edges (single incidence): handle per BC.
        let mut boundary: Vec<(usize, usize)> = Vec::new();
        for incident in map.values() {
            if incident.len() == 1 {
                boundary.push(incident[0]);
            }
        }

        match bc {
            DgBoundary::Periodic { x0, x1, y0, y1 } => {
                Self::link_periodic(geom, &tri_verts, &boundary, x0, x1, y0, y1, &mut edges)?;
            }
            DgBoundary::Compact { .. } => {
                for &(e, l) in &boundary {
                    let (a, b) = local_edge_vertices(l);
                    edges.push(EdgeLink {
                        elem: e,
                        local_edge: l,
                        own_lv: [a, b],
                        neigh: None,
                        neigh_lv: [a, b],
                        neigh_centroid: None,
                    });
                }
            }
        }
        Ok(edges)
    }

    /// Find neighbour local vertex indices `[na, nb]` whose global ids equal the
    /// owner endpoints `(own[a], own[b])`, preserving the endpoint correspondence.
    fn match_neigh(own: &[usize; 3], a: usize, b: usize, neigh: &[usize; 3]) -> [usize; 2] {
        let ga = own[a];
        let gb = own[b];
        let find = |g: usize| -> usize {
            for (k, &v) in neigh.iter().enumerate() {
                if v == g {
                    return k;
                }
            }
            0
        };
        [find(ga), find(gb)]
    }

    /// Match opposite boundary edges by translating midpoints across the domain.
    #[allow(clippy::too_many_arguments)]
    fn link_periodic(
        geom: &[ElemGeom],
        tri_verts: &[[usize; 3]],
        boundary: &[(usize, usize)],
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        edges: &mut Vec<EdgeLink>,
    ) -> PdeResult<()> {
        let lx = x1 - x0;
        let ly = y1 - y0;
        let tol = 1.0e-9 * (lx.abs() + ly.abs()).max(1.0);
        let midpoint = |e: usize, l: usize| -> [f64; 2] {
            let (a, b) = local_edge_vertices(l);
            [
                0.5 * (geom[e].p[a][0] + geom[e].p[b][0]),
                0.5 * (geom[e].p[a][1] + geom[e].p[b][1]),
            ]
        };
        for &(e, l) in boundary {
            let m = midpoint(e, l);
            // Determine which boundary this edge lies on and compute the shifted
            // target midpoint of its periodic partner.
            let on_x0 = (m[0] - x0).abs() < tol;
            let on_x1 = (m[0] - x1).abs() < tol;
            let on_y0 = (m[1] - y0).abs() < tol;
            let on_y1 = (m[1] - y1).abs() < tol;
            let mut target = m;
            if on_x0 {
                target[0] += lx;
            } else if on_x1 {
                target[0] -= lx;
            } else if on_y0 {
                target[1] += ly;
            } else if on_y1 {
                target[1] -= ly;
            } else {
                // not on a recognised boundary: treat as compact (self, no neigh)
                let (a, b) = local_edge_vertices(l);
                edges.push(EdgeLink {
                    elem: e,
                    local_edge: l,
                    own_lv: [a, b],
                    neigh: None,
                    neigh_lv: [a, b],
                    neigh_centroid: None,
                });
                continue;
            }
            // Translation that brings the partner element into this element's
            // frame so its centroid sits across edge `l` (inverse of the shift
            // that mapped `m` to `target`).
            let back_shift = [m[0] - target[0], m[1] - target[1]];
            // Locate the partner boundary edge whose midpoint ≈ target.
            let mut partner: Option<(usize, usize)> = None;
            for &(e2, l2) in boundary {
                let m2 = midpoint(e2, l2);
                if (m2[0] - target[0]).abs() < tol && (m2[1] - target[1]).abs() < tol {
                    partner = Some((e2, l2));
                    break;
                }
            }
            let (a, b) = local_edge_vertices(l);
            match partner {
                Some((e2, _l2)) => {
                    // Match neighbour locals by translated coordinates.
                    let neigh_lv = Self::match_periodic_locals(
                        geom, tri_verts, e, a, b, e2, lx, ly, on_x0, on_x1, on_y0, on_y1, tol,
                    );
                    let cn = centroid_of(&geom[e2]);
                    let neigh_centroid = [cn[0] + back_shift[0], cn[1] + back_shift[1]];
                    edges.push(EdgeLink {
                        elem: e,
                        local_edge: l,
                        own_lv: [a, b],
                        neigh: Some(e2),
                        neigh_lv,
                        neigh_centroid: Some(neigh_centroid),
                    });
                }
                None => {
                    edges.push(EdgeLink {
                        elem: e,
                        local_edge: l,
                        own_lv: [a, b],
                        neigh: None,
                        neigh_lv: [a, b],
                        neigh_centroid: None,
                    });
                }
            }
        }
        Ok(())
    }

    /// Match owner endpoints to neighbour local vertices across a periodic shift.
    #[allow(clippy::too_many_arguments)]
    fn match_periodic_locals(
        geom: &[ElemGeom],
        _tri_verts: &[[usize; 3]],
        e: usize,
        a: usize,
        b: usize,
        e2: usize,
        lx: f64,
        ly: f64,
        on_x0: bool,
        on_x1: bool,
        on_y0: bool,
        on_y1: bool,
        tol: f64,
    ) -> [usize; 2] {
        let shift = |pt: [f64; 2]| -> [f64; 2] {
            let mut s = pt;
            if on_x0 {
                s[0] += lx;
            } else if on_x1 {
                s[0] -= lx;
            } else if on_y0 {
                s[1] += ly;
            } else if on_y1 {
                s[1] -= ly;
            }
            s
        };
        let find = |pt: [f64; 2]| -> usize {
            for k in 0..3 {
                let q = geom[e2].p[k];
                if (q[0] - pt[0]).abs() < tol && (q[1] - pt[1]).abs() < tol {
                    return k;
                }
            }
            0
        };
        let pa = shift(geom[e].p[a]);
        let pb = shift(geom[e].p[b]);
        [find(pa), find(pb)]
    }

    /// Total number of DOFs (`3 * n_elem`).
    pub fn n_dofs(&self) -> usize {
        3 * self.n_elem
    }

    /// Physical coordinates of the three vertex DOFs of element `e`.
    pub fn element_vertices(&self, e: usize) -> PdeResult<[[f64; 2]; 3]> {
        if e >= self.n_elem {
            return Err(PdeError::IndexOutOfBounds {
                index: e,
                len: self.n_elem,
            });
        }
        Ok(self.geom[e].p)
    }

    /// Centroid `(x, y)` of element `e`.
    pub fn centroid(&self, e: usize) -> PdeResult<[f64; 2]> {
        if e >= self.n_elem {
            return Err(PdeError::IndexOutOfBounds {
                index: e,
                len: self.n_elem,
            });
        }
        let p = &self.geom[e].p;
        Ok([
            (p[0][0] + p[1][0] + p[2][0]) / 3.0,
            (p[0][1] + p[1][1] + p[2][1]) / 3.0,
        ])
    }

    /// Midpoint `(x, y)` of local edge `l` (opposite vertex `l`) of element `e`.
    pub fn edge_midpoint(&self, e: usize, l: usize) -> PdeResult<[f64; 2]> {
        if e >= self.n_elem {
            return Err(PdeError::IndexOutOfBounds {
                index: e,
                len: self.n_elem,
            });
        }
        let (a, b) = local_edge_vertices(l);
        let p = &self.geom[e].p;
        Ok([0.5 * (p[a][0] + p[b][0]), 0.5 * (p[a][1] + p[b][1])])
    }

    /// Area of element `e`.
    pub fn area(&self, e: usize) -> PdeResult<f64> {
        if e >= self.n_elem {
            return Err(PdeError::IndexOutOfBounds {
                index: e,
                len: self.n_elem,
            });
        }
        Ok(self.geom[e].area)
    }

    /// Neighbour element index across each local edge `l` (0..3) of element `e`,
    /// or `None` for an unmatched boundary edge. Used by the slope limiter.
    pub fn neighbors(&self, e: usize) -> PdeResult<[Option<usize>; 3]> {
        if e >= self.n_elem {
            return Err(PdeError::IndexOutOfBounds {
                index: e,
                len: self.n_elem,
            });
        }
        let mut out = [None; 3];
        for link in &self.edges {
            if link.elem == e {
                out[link.local_edge] = link.neigh;
            }
        }
        Ok(out)
    }

    /// Per local edge `l` (0..3) of element `e`: the neighbour element index and
    /// the neighbour centroid expressed in `e`'s frame (periodic shift applied),
    /// or `None` for an unmatched boundary edge. Used by the geometry-aware
    /// slope limiter to estimate directional cell-mean gradients.
    pub fn limiter_stencil(&self, e: usize) -> PdeResult<LimiterStencil> {
        if e >= self.n_elem {
            return Err(PdeError::IndexOutOfBounds {
                index: e,
                len: self.n_elem,
            });
        }
        let mut out: LimiterStencil = [None, None, None];
        for link in &self.edges {
            if link.elem == e {
                if let (Some(ne), Some(c)) = (link.neigh, link.neigh_centroid) {
                    out[link.local_edge] = Some((ne, c));
                }
            }
        }
        Ok(out)
    }

    /// Cell mean of the P1 field on element `e` (`(u0+u1+u2)/3` for nodal P1).
    pub fn cell_mean(&self, u: &[f64], e: usize) -> f64 {
        let base = 3 * e;
        (u[base] + u[base + 1] + u[base + 2]) / 3.0
    }

    /// Total discrete mass `Σ_T ∫_T u_h` (uses `∫_T u = |T|·mean`).
    pub fn total_mass(&self, u: &[f64]) -> f64 {
        let mut m = 0.0;
        for e in 0..self.n_elem {
            m += self.geom[e].area * self.cell_mean(u, e);
        }
        m
    }

    /// Maximum stable advection time step from the CFL condition
    /// `dt ≤ C · h_min / |β|` with a conservative constant for P1 (`C≈1/3`).
    pub fn cfl_dt(&self, bx: f64, by: f64, courant: f64) -> f64 {
        let speed = (bx * bx + by * by).sqrt().max(1.0e-30);
        let mut h_min = f64::INFINITY;
        for g in &self.geom {
            // characteristic length ~ smallest altitude ≈ 2*area / longest edge
            let e01 = dist(g.p[0], g.p[1]);
            let e12 = dist(g.p[1], g.p[2]);
            let e20 = dist(g.p[2], g.p[0]);
            let longest = e01.max(e12).max(e20);
            let alt = 2.0 * g.area / longest;
            if alt < h_min {
                h_min = alt;
            }
        }
        courant * h_min / speed
    }
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn centroid_of(g: &ElemGeom) -> [f64; 2] {
    [
        (g.p[0][0] + g.p[1][0] + g.p[2][0]) / 3.0,
        (g.p[0][1] + g.p[1][1] + g.p[2][1]) / 3.0,
    ]
}

/// Physical flux `f(u) = (fx, fy)` for the chosen model.
fn flux_vec(flux: DgFlux, u: f64) -> [f64; 2] {
    match flux {
        DgFlux::Advection { bx, by } => [bx * u, by * u],
        DgFlux::Burgers => [0.5 * u * u, 0.0],
    }
}

/// Maximum absolute normal wave speed `|∂(f·n)/∂u|` over the trace states, used
/// as the Lax-Friedrichs dissipation coefficient.
fn max_normal_speed(flux: DgFlux, n: [f64; 2], ul: f64, ur: f64) -> f64 {
    match flux {
        DgFlux::Advection { bx, by } => (bx * n[0] + by * n[1]).abs(),
        DgFlux::Burgers => (ul * n[0]).abs().max((ur * n[0]).abs()),
    }
}

/// Numerical normal flux `f̂·n` for left state `ul` (interior) and right `ur`
/// (exterior) across a face with outward normal `n`.
///
/// Advection uses exact upwinding; Burgers uses local Lax-Friedrichs (Rusanov).
fn numerical_normal_flux(flux: DgFlux, n: [f64; 2], ul: f64, ur: f64) -> f64 {
    match flux {
        DgFlux::Advection { bx, by } => {
            let bn = bx * n[0] + by * n[1];
            if bn >= 0.0 { bn * ul } else { bn * ur }
        }
        DgFlux::Burgers => {
            let fl = flux_vec(flux, ul);
            let fr = flux_vec(flux, ur);
            let fln = fl[0] * n[0] + fl[1] * n[1];
            let frn = fr[0] * n[0] + fr[1] * n[1];
            let alpha = max_normal_speed(flux, n, ul, ur);
            0.5 * (fln + frn) - 0.5 * alpha * (ur - ul)
        }
    }
}

/// 2-point Gauss-Legendre nodes/weights on `[0,1]` (exact for cubics).
const GAUSS2_PTS: [f64; 2] = [
    0.5 - 0.5 / 1.732_050_807_568_877_2,
    0.5 + 0.5 / 1.732_050_807_568_877_2,
];
const GAUSS2_W: [f64; 2] = [0.5, 0.5];

/// Evaluate the right-hand side `M_T⁻¹ (V_T − S_T)` for every element, returning
/// the nodal time-derivative `du/dt` (length `3*n_elem`).
fn dg_rhs(space: &Dg2dSpace, u: &[f64], flux: DgFlux, bc: DgBoundary) -> PdeResult<Vec<f64>> {
    let n = space.n_dofs();
    if u.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u.len()],
        });
    }
    // Residual r_i = V_i − S_i accumulated per element (before mass inverse).
    let mut res = vec![0.0_f64; n];

    // Volume term: V_i = ∫_T f(u)·∇λ_i. With u linear, f(u) is not linear in
    // general (Burgers), so use a 3-point (centroid-edge) rule, exact for the
    // advection case and 2nd-order for Burgers (sufficient with the limiter).
    for e in 0..space.n_elem {
        let g = &space.geom[e];
        let base = 3 * e;
        let u0 = u[base];
        let u1 = u[base + 1];
        let u2 = u[base + 2];
        // V_i = ∫_T f(u)·∇λ_i = (∫_T f)·∇λ_i since ∇λ_i is constant on T.
        // Integrate f by the 3-edge-midpoint rule (weight |T|/3 each), exact for
        // quadratics — hence exact for advection (f linear) and 2nd-order for
        // Burgers (f quadratic). Midpoint of edge opposite v_q has λ_q=0.
        let um = [0.5 * (u1 + u2), 0.5 * (u2 + u0), 0.5 * (u0 + u1)];
        let w = g.area / 3.0;
        let mut int_f = [0.0_f64; 2];
        for &uq in &um {
            let fq = flux_vec(flux, uq);
            int_f[0] += w * fq[0];
            int_f[1] += w * fq[1];
        }
        for i in 0..3 {
            res[base + i] += int_f[0] * g.grad[i][0] + int_f[1] * g.grad[i][1];
        }
    }

    // Surface term: subtract S_i = ∮_∂T (f̂·n) λ_i ds, integrated per edge.
    for link in &space.edges {
        let e = link.elem;
        let g = &space.geom[e];
        let l = link.local_edge;
        let (pa, pb) = (g.p[link.own_lv[0]], g.p[link.own_lv[1]]);
        let elen = dist(pa, pb);
        // outward normal on this edge (rotate tangent, orient away from v_l).
        let n = outward_edge_normal(g, l);
        let base = 3 * e;
        let (oa, ob) = (link.own_lv[0], link.own_lv[1]);
        // interior nodal values at the two edge endpoints
        let ula = u[base + oa];
        let ulb = u[base + ob];
        // exterior nodal values at the matching endpoints
        let (ura, urb) = match link.neigh {
            Some(ne) => {
                let nbase = 3 * ne;
                (u[nbase + link.neigh_lv[0]], u[nbase + link.neigh_lv[1]])
            }
            None => match bc {
                DgBoundary::Compact { far_field } => (far_field, far_field),
                // periodic-unmatched fallback: reflect interior (no flux contribution bias)
                DgBoundary::Periodic { .. } => (ula, ulb),
            },
        };
        // 2-point Gauss along the edge; param s∈[0,1] from endpoint a→b.
        for q in 0..2 {
            let s = GAUSS2_PTS[q];
            let wq = GAUSS2_W[q] * elen;
            let ul = ula * (1.0 - s) + ulb * s;
            let ur = ura * (1.0 - s) + urb * s;
            let fhat = numerical_normal_flux(flux, n, ul, ur);
            // test functions along the edge: λ at endpoint a = (1-s), at b = s,
            // and the off-edge vertex λ = 0.
            // contribution to res at local node oa and ob.
            res[base + oa] -= wq * fhat * (1.0 - s);
            res[base + ob] -= wq * fhat * s;
        }
    }

    // Apply the inverse P1 mass matrix per element: du/dt = M_T⁻¹ res.
    let mut dudt = vec![0.0_f64; n];
    for e in 0..space.n_elem {
        let area = space.geom[e].area;
        let base = 3 * e;
        let r = [res[base], res[base + 1], res[base + 2]];
        let m_inv_r = p1_mass_solve(area, r);
        dudt[base] = m_inv_r[0];
        dudt[base + 1] = m_inv_r[1];
        dudt[base + 2] = m_inv_r[2];
    }
    Ok(dudt)
}

/// Outward unit normal on local edge `l` (opposite vertex `l`) of element geom.
fn outward_edge_normal(g: &ElemGeom, l: usize) -> [f64; 2] {
    let (a, b) = local_edge_vertices(l);
    let tx = g.p[b][0] - g.p[a][0];
    let ty = g.p[b][1] - g.p[a][1];
    let len = (tx * tx + ty * ty).sqrt();
    let mut nx = ty / len;
    let mut ny = -tx / len;
    let mx = 0.5 * (g.p[a][0] + g.p[b][0]);
    let my = 0.5 * (g.p[a][1] + g.p[b][1]);
    let ox = mx - g.p[l][0];
    let oy = my - g.p[l][1];
    if nx * ox + ny * oy < 0.0 {
        nx = -nx;
        ny = -ny;
    }
    [nx, ny]
}

/// Solve `M r = rhs` for the P1 mass matrix `M = (area/12)[[2,1,1],[1,2,1],[1,1,2]]`.
///
/// Closed-form inverse: `M⁻¹ = (3/area)·([[3,-1,-1],[-1,3,-1],[-1,-1,3]])/? `.
/// We use the exact inverse of `[[2,1,1],[1,2,1],[1,1,2]]`, which is
/// `(1/4)[[3,-1,-1],[-1,3,-1],[-1,-1,3]]`, scaled by `12/area`.
fn p1_mass_solve(area: f64, rhs: [f64; 3]) -> [f64; 3] {
    // M = (area/12) * A0,  A0 = [[2,1,1],[1,2,1],[1,1,2]].
    // A0^{-1} = (1/4) [[3,-1,-1],[-1,3,-1],[-1,-1,3]].
    // M^{-1} = (12/area) * A0^{-1} = (3/area) [[3,-1,-1],[-1,3,-1],[-1,-1,3]].
    let s = 3.0 / area;
    [
        s * (3.0 * rhs[0] - rhs[1] - rhs[2]),
        s * (-rhs[0] + 3.0 * rhs[1] - rhs[2]),
        s * (-rhs[0] - rhs[1] + 3.0 * rhs[2]),
    ]
}

/// One SSP-RK3 (Shu-Osher) stage update of the DG semi-discretisation, with an
/// optional per-stage limiter applied to the intermediate states.
fn ssp_rk3_step(
    space: &Dg2dSpace,
    u: &mut [f64],
    flux: DgFlux,
    bc: DgBoundary,
    dt: f64,
    limiter: Option<LimiterHook<'_>>,
) -> PdeResult<()> {
    let n = u.len();
    // Stage 1: u1 = u + dt L(u)
    let k0 = dg_rhs(space, u, flux, bc)?;
    let mut u1 = vec![0.0_f64; n];
    for i in 0..n {
        u1[i] = u[i] + dt * k0[i];
    }
    if let Some(lim) = limiter {
        lim(space, &mut u1)?;
    }
    // Stage 2: u2 = 3/4 u + 1/4 (u1 + dt L(u1))
    let k1 = dg_rhs(space, &u1, flux, bc)?;
    let mut u2 = vec![0.0_f64; n];
    for i in 0..n {
        u2[i] = 0.75 * u[i] + 0.25 * (u1[i] + dt * k1[i]);
    }
    if let Some(lim) = limiter {
        lim(space, &mut u2)?;
    }
    // Stage 3: u = 1/3 u + 2/3 (u2 + dt L(u2))
    let k2 = dg_rhs(space, &u2, flux, bc)?;
    for i in 0..n {
        u[i] = (1.0 / 3.0) * u[i] + (2.0 / 3.0) * (u2[i] + dt * k2[i]);
    }
    if let Some(lim) = limiter {
        lim(space, u)?;
    }
    Ok(())
}

/// Advect an initial nodal field `u0` with the linear-advection DG scheme.
///
/// # Arguments
/// * `mesh` — triangular mesh.
/// * `u0` — initial nodal P1 values (`3*n_elem`, vertex-ordered per element).
/// * `beta` — constant advection velocity `(βx, βy)`.
/// * `dt`, `nsteps` — explicit time step and number of SSP-RK3 steps.
/// * `bc` — boundary treatment.
/// * `limiter` — when `true`, apply the minmod slope limiter each RK stage.
///
/// # Returns
/// The nodal field after `nsteps` steps (length `3*n_elem`).
///
/// # Errors
/// Returns [`PdeError::CflViolation`] if `dt` exceeds the advection CFL limit.
pub fn dg_2d_advect(
    mesh: &TriMesh2d,
    u0: &[f64],
    beta: (f64, f64),
    dt: f64,
    nsteps: usize,
    bc: DgBoundary,
    limiter: bool,
) -> PdeResult<Vec<f64>> {
    let space = Dg2dSpace::new(mesh, bc)?;
    let n = space.n_dofs();
    if u0.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u0.len()],
        });
    }
    let dt_max = space.cfl_dt(beta.0, beta.1, 1.0);
    if dt > dt_max {
        return Err(PdeError::CflViolation { dt, dt_max });
    }
    let flux = DgFlux::Advection {
        bx: beta.0,
        by: beta.1,
    };
    let mut u = u0.to_vec();
    let gmin = u0.iter().cloned().fold(f64::INFINITY, f64::min);
    let gmax = u0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lim_closure = crate::dg::limiter_2d::minmod_bounded_closure(gmin, gmax);
    let lim: Option<LimiterHook<'_>> = if limiter { Some(&lim_closure) } else { None };
    for _ in 0..nsteps {
        ssp_rk3_step(&space, &mut u, flux, bc, dt, lim)?;
    }
    Ok(u)
}

/// Evolve an initial nodal field with the inviscid-Burgers DG scheme (flux
/// `u²/2` along x). Always limited to remain monotone across shocks.
///
/// # Arguments mirror [`dg_2d_advect`]; `max_speed` is the largest `|u|` used for
/// the CFL estimate. Returns the nodal field after `nsteps` steps.
pub fn dg_2d_burgers(
    mesh: &TriMesh2d,
    u0: &[f64],
    max_speed: f64,
    dt: f64,
    nsteps: usize,
    bc: DgBoundary,
    limiter: bool,
) -> PdeResult<Vec<f64>> {
    let space = Dg2dSpace::new(mesh, bc)?;
    let n = space.n_dofs();
    if u0.len() != n {
        return Err(PdeError::ShapeMismatch {
            expected: vec![n],
            got: vec![u0.len()],
        });
    }
    let dt_max = space.cfl_dt(max_speed.abs().max(1.0e-12), 0.0, 1.0);
    if dt > dt_max {
        return Err(PdeError::CflViolation { dt, dt_max });
    }
    let flux = DgFlux::Burgers;
    let mut u = u0.to_vec();
    let gmin = u0.iter().cloned().fold(f64::INFINITY, f64::min);
    let gmax = u0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lim_closure = crate::dg::limiter_2d::minmod_bounded_closure(gmin, gmax);
    let lim: Option<LimiterHook<'_>> = if limiter { Some(&lim_closure) } else { None };
    for _ in 0..nsteps {
        ssp_rk3_step(&space, &mut u, flux, bc, dt, lim)?;
    }
    Ok(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(n: usize) -> TriMesh2d {
        TriMesh2d::rect_grid(0.0, 1.0, 0.0, 1.0, n, n).expect("mesh")
    }

    fn periodic_unit() -> DgBoundary {
        DgBoundary::Periodic {
            x0: 0.0,
            x1: 1.0,
            y0: 0.0,
            y1: 1.0,
        }
    }

    /// Initialise nodal P1 DOFs from a function evaluated at vertices.
    fn init_nodal<F: Fn(f64, f64) -> f64>(space: &Dg2dSpace, f: F) -> Vec<f64> {
        let mut u = vec![0.0; space.n_dofs()];
        for e in 0..space.n_elem {
            let v = space.element_vertices(e).expect("v");
            for i in 0..3 {
                u[3 * e + i] = f(v[i][0], v[i][1]);
            }
        }
        u
    }

    #[test]
    fn mass_matrix_inverse_correct() {
        // M * (M^{-1} rhs) == rhs.
        let area = 0.37;
        let rhs = [1.3, -0.7, 2.1];
        let x = p1_mass_solve(area, rhs);
        // M x:
        let s = area / 12.0;
        let mx = [
            s * (2.0 * x[0] + x[1] + x[2]),
            s * (x[0] + 2.0 * x[1] + x[2]),
            s * (x[0] + x[1] + 2.0 * x[2]),
        ];
        for i in 0..3 {
            assert!((mx[i] - rhs[i]).abs() < 1e-12, "{} != {}", mx[i], rhs[i]);
        }
    }

    #[test]
    fn periodic_edges_all_matched() {
        let mesh = square(5);
        let space = Dg2dSpace::new(&mesh, periodic_unit()).expect("ok");
        // Every edge link must have a neighbour under periodic BC.
        for link in &space.edges {
            assert!(link.neigh.is_some(), "unmatched edge in periodic mesh");
        }
    }

    #[test]
    fn constant_field_is_steady() {
        // A constant field must be advected exactly (zero residual).
        let mesh = square(5);
        let u0 = {
            let space = Dg2dSpace::new(&mesh, periodic_unit()).expect("ok");
            init_nodal(&space, |_, _| 2.5)
        };
        let dt = {
            let space = Dg2dSpace::new(&mesh, periodic_unit()).expect("ok");
            0.5 * space.cfl_dt(1.0, 0.5, 1.0)
        };
        let u = dg_2d_advect(&mesh, &u0, (1.0, 0.5), dt, 20, periodic_unit(), false).expect("ok");
        for &v in &u {
            assert!((v - 2.5).abs() < 1e-11, "constant not preserved: {v}");
        }
    }

    #[test]
    fn mass_conserved_periodic() {
        // Σ_T ∫_T u_h constant in time to ~1e-12 with periodic BC.
        let mesh = square(9);
        let space = Dg2dSpace::new(&mesh, periodic_unit()).expect("ok");
        let u0 = init_nodal(&space, |x, y| {
            let dx = x - 0.5;
            let dy = y - 0.5;
            (-40.0 * (dx * dx + dy * dy)).exp()
        });
        let m0 = space.total_mass(&u0);
        let dt = 0.4 * space.cfl_dt(1.0, 0.7, 1.0);
        let u = dg_2d_advect(&mesh, &u0, (1.0, 0.7), dt, 30, periodic_unit(), false).expect("ok");
        let m1 = space.total_mass(&u);
        assert!((m1 - m0).abs() < 1e-12, "mass drift {} -> {}", m0, m1);
    }

    #[test]
    fn cfl_violation_detected() {
        let mesh = square(5);
        let space = Dg2dSpace::new(&mesh, periodic_unit()).expect("ok");
        let u0 = init_nodal(&space, |_, _| 1.0);
        let dt_bad = 10.0 * space.cfl_dt(1.0, 0.0, 1.0);
        let r = dg_2d_advect(&mesh, &u0, (1.0, 0.0), dt_bad, 1, periodic_unit(), false);
        assert!(matches!(r, Err(PdeError::CflViolation { .. })));
    }

    #[test]
    fn smooth_gaussian_one_period_returns() {
        // Advect a smooth Gaussian by β=(1,0) for one full period (T=1 on unit
        // domain) WITHOUT limiter; should return close to itself (high order).
        let mesh = square(13);
        let space = Dg2dSpace::new(&mesh, periodic_unit()).expect("ok");
        let g0 = |x: f64, y: f64| {
            // periodic-friendly smooth bump using cos² in x, gaussian in y band
            let dx = x - 0.5;
            let dy = y - 0.5;
            (-30.0 * (dx * dx + dy * dy)).exp()
        };
        let u0 = init_nodal(&space, g0);
        let beta = (1.0, 0.0);
        let dt = 0.3 * space.cfl_dt(beta.0, beta.1, 1.0);
        let nsteps = (1.0 / dt).ceil() as usize;
        let dt_exact = 1.0 / nsteps as f64; // land exactly on T=1
        let u =
            dg_2d_advect(&mesh, &u0, beta, dt_exact, nsteps, periodic_unit(), false).expect("ok");
        // L2 error vs initial (one period ⇒ identity for exact advection).
        let mut err2 = 0.0;
        let mut nrm2 = 0.0;
        for e in 0..space.n_elem {
            let area = space.area(e).expect("a");
            for i in 0..3 {
                let d = u[3 * e + i] - u0[3 * e + i];
                err2 += area / 3.0 * d * d;
                nrm2 += area / 3.0 * u0[3 * e + i] * u0[3 * e + i];
            }
        }
        let rel = (err2 / nrm2).sqrt();
        assert!(rel < 0.15, "one-period L2 error too large: {rel}");
    }

    #[test]
    fn mass_conserved_with_limiter() {
        // The minmod + MPP limiter preserves every cell mean, so total mass is
        // still conserved to round-off even on a discontinuous profile.
        let mesh = square(15);
        let bc = periodic_unit();
        let space = Dg2dSpace::new(&mesh, bc).expect("ok");
        let u0 = init_nodal(&space, |x, _| if x < 0.5 { 1.0 } else { 0.0 });
        let m0 = space.total_mass(&u0);
        let dt = 0.3 * space.cfl_dt(1.0, 0.0, 1.0);
        let u = dg_2d_advect(&mesh, &u0, (1.0, 0.0), dt, 25, bc, true).expect("ok");
        let m1 = space.total_mass(&u);
        assert!(
            (m1 - m0).abs() < 1e-12,
            "mass drift with limiter {m0} -> {m1}"
        );
    }

    #[test]
    fn burgers_shock_travels_at_rh_speed() {
        // Riemann step uL=1 (x<1), uR=0 (x>1) on x∈[-1,3]. RH speed s=(1+0)/2=0.5.
        let nx = 81;
        let ny = 3;
        let mesh = TriMesh2d::rect_grid(-1.0, 3.0, 0.0, 0.1, nx, ny).expect("mesh");
        let bc = DgBoundary::Compact { far_field: 0.0 };
        let space = Dg2dSpace::new(&mesh, bc).expect("ok");
        let u0 = init_nodal(&space, |x, _| if x < 1.0 { 1.0 } else { 0.0 });
        let t_final = 1.0;
        let dt = 0.4 * space.cfl_dt(1.0, 0.0, 1.0);
        let nsteps = (t_final / dt).ceil() as usize;
        let dte = t_final / nsteps as f64;
        let u = dg_2d_burgers(&mesh, &u0, 1.0, dte, nsteps, bc, true).expect("ok");
        // Find front: largest x where cell mean > 0.5.
        let mut front = -1.0_f64;
        for e in 0..space.n_elem {
            let c = space.centroid(e).expect("c");
            let m = space.cell_mean(&u, e);
            if m > 0.5 && c[0] > front {
                front = c[0];
            }
        }
        let analytic = 1.0 + 0.5 * t_final;
        println!("BURGERS front={front} analytic={analytic}");
        assert!(
            (front - analytic).abs() < 0.1,
            "front {front} vs {analytic}"
        );
    }

    #[test]
    fn step_stays_monotone_with_limiter() {
        // A step profile advected with the limiter ON must stay within [0,1]
        // (no new extrema); without it, overshoots appear.
        let mesh = square(21);
        let bc = periodic_unit();
        let space = Dg2dSpace::new(&mesh, bc).expect("ok");
        let u0 = init_nodal(&space, |x, _| if x < 0.5 { 1.0 } else { 0.0 });
        let beta = (1.0, 0.0);
        let dt = 0.3 * space.cfl_dt(beta.0, beta.1, 1.0);
        let limited = dg_2d_advect(&mesh, &u0, beta, dt, 40, bc, true).expect("ok");
        let unlimited = dg_2d_advect(&mesh, &u0, beta, dt, 40, bc, false).expect("ok");
        let umax_lim = limited.iter().cloned().fold(f64::MIN, f64::max);
        let umin_lim = limited.iter().cloned().fold(f64::MAX, f64::min);
        let umax_unl = unlimited.iter().cloned().fold(f64::MIN, f64::max);
        let umin_unl = unlimited.iter().cloned().fold(f64::MAX, f64::min);
        println!("LIMITED [{umin_lim},{umax_lim}]  UNLIMITED [{umin_unl},{umax_unl}]");
        assert!(umax_lim < 1.0 + 1e-9, "limited overshoot: {umax_lim}");
        assert!(umin_lim > -1e-9, "limited undershoot: {umin_lim}");
        // The unlimited scheme should overshoot the [0,1] bound on this step.
        assert!(
            umax_unl > 1.0 + 1e-3 || umin_unl < -1e-3,
            "no overshoot seen"
        );
    }
}
