# Code Reading Comments

This document is the detailed annotation layer for the copied BP3333 virtual
C1 implementation.  It explains every Python submodule and every class/function
in the folder, so the numerical code can be read without first reverse
engineering the call graph.

## `__init__.py`

Module role: package surface. It re-exports the main data classes and workflow
functions that are useful to external scripts.

`BP3333EllipseParameters`: immutable parameter container for the BP3333
ellipse/virtual-thickness model.

`DirectFitResult`: result object returned by direct parameter extraction.

`OptimizedFitResult`: result object returned after optimisation.

`build_direct_fit`: direct no-optimisation reconstruction from one airfoil file.

`generate_airfoil`: forward geometry generator from BP3333 parameters.

`optimize_fit`: optimisation wrapper around the direct result.

## `bezier.py`

Module role: small Bezier utilities used by the thickness and camber builders.

`cubic_bezier(control, u)`: evaluates a cubic Bezier curve from four scalar
control values. The same function is used for x-control and y-control arrays.

`cosine_sine_parameters(n)`: creates clustered Bezier parameters. Cosine
spacing improves leading-edge resolution; sine spacing improves trailing-edge
resolution.

## `ellipse.py`

Module role: leading and trailing ellipse utilities. The leading ellipse is
anchored at the thickness nose and provides the front head before the BP3333
Bezier body.

`EllipseHead`: dataclass storing leading ellipse semi-axes and tangent
parameter.

`EllipseHead.x_tangent`: x-coordinate of the leading ellipse/Bezier junction.

`EllipseHead.y_tangent`: thickness value at the leading ellipse/Bezier junction.

`EllipseHead.slope_tangent`: dy/dx at the leading ellipse tangent point.

`fit_ellipse_head`: locally fits the leading ellipse size to reference
thickness data near the nose.

`make_ellipse_head`: validates ellipse axes and clips the tangent parameter so
the ellipse joins before maximum thickness.

`sample_ellipse`: samples the leading ellipse from the nose to its tangent
point.

`fit_trailing_ellipse`: locally fits a trailing ellipse seed through the
trailing thickness endpoint.

`trailing_theta_from_beta`: estimates trailing ellipse tangent parameter from
the trailing wedge angle.

## `geometry.py`

Module role: input airfoil preprocessing. It reads coordinate files, splits
upper/lower surfaces, derives camber/thickness distributions, and extracts
initial BP3333 parameters.

`ReferenceAirfoil`: dataclass containing original contour, reference evaluation
grid, upper/lower/camber/thickness arrays, splines, and native distributions.

`read_airfoil`: loads an airfoil file and builds a `ReferenceAirfoil` object.

`extract_seed`: extracts direct initial parameters such as `x_t`, `y_t`,
curvatures, camber extrema, and edge angles.

`_estimate_trailing_edge_values`: estimates trailing-edge camber offset and
half-thickness, with special handling for closed blunt trailing edges.

`_normalise_chord`: translates/scales input coordinates to unit chord.

`_unique_xy`: removes repeated x entries by averaging y values.

`_unique_columns`: generalised unique-x helper for multiple y-like columns.

`_reference_distributions_python_par`: reproduces the original python-par
camber/thickness extraction convention.

`_fitpack_interp`: cubic spline interpolation helper using SciPy FITPACK.

`_pchip_from_xy`: monotone-safe PCHIP constructor for reference surfaces.

`_safe_derivative`: finite-difference derivative helper.

`_local_second_derivative`: local quadratic estimate of curvature-like second
derivative near a target index.

`_tail_slope`: robust linear slope estimate near the trailing edge.

`_nose_slope`: robust linear slope estimate near the leading edge.

## `fit.py`

Module role: direct construction before global/local optimisation.

`DirectFitResult`: dataclass bundling direct parameters, generated geometry,
reference data, and error metrics.

`build_direct_fit`: builds one direct BP3333 virtual-C1 geometry from an airfoil
file and chosen ellipse/virtual-thickness settings.

`_result_from_params`: internal helper that evaluates geometry and metrics from
already assembled parameters.

`_unique_values`: removes duplicate sweep values while preserving deterministic
ordering.

`_is_continuous_result`: checks whether a direct fit has acceptable virtual
control continuity diagnostics.

`sweep_direct_fit`: searches theta/axis-ratio/virtual-thickness seeds and keeps
the best acceptable direct fit.

`surface_errors`: evaluates model upper/lower y-errors on the reference x grid.

`_safe_surface_spline`: sorted unique-x interpolation helper for possibly
non-uniform generated surfaces.

## `model.py`

Module role: mathematical BP3333 forward model. This is the core of the
parameterisation.

`BP3333EllipseParameters`: immutable parameter set for thickness, camber,
ellipse heads, virtual thicknesses, and optional trailing C2 behaviour.

`BP3333EllipseParameters.to_dict`: serialises parameters into JSON-friendly
scalars.

`TailEllipse`: dataclass for optional trailing ellipse.

`TailEllipse.x_tangent`: x-coordinate of the trailing ellipse tangent point.

`TailEllipse.y_tangent`: thickness at the trailing ellipse tangent point.

`TailEllipse.slope_tangent`: dy/dx at the trailing ellipse tangent point.

`TailEllipse.curvature`: signed curvature of the trailing ellipse at tangent.

`ThicknessControls`: dataclass storing all thickness control points and
diagnostics.

`generate_airfoil`: full forward pass from parameters to thickness, camber, and
upper/lower surfaces.

`thickness_distribution`: samples leading ellipse, leading Bezier, trailing
Bezier, and optional trailing ellipse into one thickness curve.

`thickness_control_points`: builds the BP3333 thickness control polygon from
ellipse tangent data and virtual-C1 equations.

`ellipse_curvature`: signed curvature of the leading ellipse at the tangent.

`solve_virtual_c1_controls`: solves the virtual-thickness control construction
used to bridge the ellipse and Bezier body.

`_mt_g2_seed_vectors`: returns seed vectors used when solving maximum-thickness
G2-style compatibility equations.

`_maximum_thickness_curvature`: evaluates the cubic curvature at maximum
thickness.

`solve_leading_virtual_controls`: solves leading-side virtual controls.

`solve_trailing_virtual_controls`: solves trailing-side virtual controls.

`_cubic_derivative`: first derivative of a cubic Bezier in parameter space.

`_cubic_second_derivative`: second derivative of a cubic Bezier in parameter
space.

`_cubic_curvature`: converts parametric first/second derivatives into signed
curvature.

`trailing_ellipse_enabled`: returns whether the optional trailing ellipse is
active.

`make_tail_ellipse`: creates or solves the trailing ellipse tangent state.

`solve_tail_theta_c2`: solves trailing ellipse tangent parameter when C2 is
requested.

`tail_c2_residual`: residual of the trailing Bezier/ellipse curvature-matching
condition.

`sample_trailing_ellipse`: samples the trailing ellipse from tangent to TE.

`solve_q_from_de_formula`: solves the q-root formula inherited from the
BP3333/ellipse derivation.

`camber_distribution`: samples the BP3333 camber curve.

`camber_control_points`: constructs leading/trailing cubic camber control
points.

`solve_b1`: solves the central camber control value from curvature and edge
angle constraints.

`thickness_camber_to_surfaces`: offsets camber by normal thickness to generate
upper/lower surfaces.

`_validate_distribution`: verifies monotone/finite generated distributions.

`_validate_control_polygon`: checks Bezier x-control ordering.

`_positive_sqrt`: safe square-root helper for formulas with small roundoff.

`_hermite_camber`: fallback cubic Hermite camber line for difficult parameter
sets.

## `optimization.py`

Module role: turns a direct fit into an optimised fit. It defines the design
vector, bounds, constraints, objective, SLSQP, and optional built-in GA.

`VARIABLE_NAMES`: ordered names of the optimisation vector.

`OptimizedFitResult`: dataclass storing initial/optimised results and optimiser
diagnostics.

`_GAResult`: compact internal result for the real-coded GA.

`optimize_fit`: top-level optimisation entry.

`_make_feasible_optimized_result`: fallback result creation when a candidate
vector is not better or not feasible.

`_slsqp_optimize`: local constrained SLSQP call.

`_ga_optimize`: simple real-coded genetic algorithm.

`_tournament`: GA parent selection helper.

`_constraint_violation`: converts inequality constraint violations into scalar
penalty.

`params_to_vector`: serialises `BP3333EllipseParameters` to optimisation vector.

`vector_to_params`: converts an optimisation vector back into parameters.

`build_bounds`: creates physical/local bounds around the direct estimate.

`build_constraints`: creates scipy inequality constraints for control ordering
and continuity feasibility.

`build_objective`: constructs the chosen fitting objective (`mae`, `rms`, or
BP3434-style surface L2).

`make_result_from_vector`: evaluates a vector and returns a `DirectFitResult`
shaped object.

`_surface_errors_on_reference_grid`: delegates reference-grid error evaluation.

`_continuous_enough`: continuity diagnostic for accepting an optimised result.

`_negative_curvature_bounds`: creates stable negative curvature bounds.

`_has_auxiliary_closed_trailing_edge`: detects duplicate closed blunt TE input.

`_tail_thickness_points`: selects tail reference thickness points for optional
tail penalty.

`_surface_outside_reference`: rejects generated surfaces outside reference x
range.

## `io.py`

Module role: serialisation of coordinates, parameters, optimisation diagnostics,
and batch summaries.

`save_coordinates`: writes reconstructed upper/lower coordinates to a DAT file.

`save_parameters`: writes direct-fit parameters and metrics to JSON.

`save_optimization_parameters`: writes initial/optimised variables, controls,
and metrics to JSON.

`_tail_payload`: serialises optional trailing ellipse diagnostics.

`save_summary`: writes batch metric rows to CSV.

## `plotting.py`

Module role: diagnostic visualisation for direct and optimised reconstructions.

`plot_result`: plots reference/model airfoil and thickness/camber diagnostics
for a direct result.

`plot_optimization_result`: plots before/after optimisation comparisons.

`_plot_thickness`: draws thickness, control points, and ellipse segments.

`_plot_controls`: draws Bezier control polygons with labelled points.

`_plot_airfoil`: draws reference and reconstructed airfoil surfaces.

`_plot_optimized_thickness`: before/after thickness comparison for optimisation.

`_plot_optimized_airfoil`: before/after airfoil comparison for optimisation.

`_tail_ellipse_xy`: extracts optional trailing ellipse plot coordinates.

## `main.py`

Module role: shared workflow called by both entry points.

`run_one`: reconstructs one airfoil and writes selected outputs.

`run_all`: loops over every airfoil in a test directory and writes a summary.

`main`: legacy CLI-compatible entry retained for backward compatibility; new
terminal runs should prefer `main_cli.py`.

`_resolve_path`: resolves absolute, relative, and package-local paths.

`_parse_sweep`: parses sweep strings such as `0.4:1.45:22` or comma lists.

## `main_cli.py`

Module role: command-line-only entry point.

`build_parser`: constructs the argparse option set.

`main`: parses CLI options and dispatches to `run_one` or `run_all`.

## `main_spyder.py`

Module role: Spyder-friendly direct-run entry point.

`main`: reads the constants in `USER SETTINGS FOR SPYDER` and dispatches to
`run_one` or `run_all` without argparse.
