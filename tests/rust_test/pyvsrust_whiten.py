# %%
import cProfile
import pstats
import time
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np

import dspant.processors.spatial.whiten as py_whiten

# %%


def detailed_performance_comparison(size=1000, repeats=10):
    """
    Detailed performance analysis of whitening methods
    """
    # Create a random data matrix
    np.random.seed(42)
    data = np.random.randn(size, size)

    # Python implementation
    py_times = []
    py_profiles = []
    for _ in range(repeats):
        # Profiling wrapper
        pr = cProfile.Profile()
        pr.enable()
        start = time.time()

        # Use Python whitening processor
        py_processor = py_whiten.WhiteningProcessor(mode="global")
        py_result = py_processor._compute_whitening_from_covariance(
            data.T @ data / len(data), eps=1e-6
        )

        py_time = time.time() - start
        pr.disable()

        # Capture profiling output
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats()

        py_times.append(py_time)
        py_profiles.append(s.getvalue())

    # Rust implementation
    try:
        from dspant._rs import compute_whitening_matrix

        rs_times = []
        rs_profiles = []
        for _ in range(repeats):
            # Profiling wrapper
            pr = cProfile.Profile()
            pr.enable()
            start = time.time()

            # Use Rust whitening function
            rs_result = compute_whitening_matrix(
                (data.T @ data / len(data)).astype(np.float32), 1e-6
            )

            rs_time = time.time() - start
            pr.disable()

            # Capture profiling output
            s = StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
            ps.print_stats()

            rs_times.append(rs_time)
            rs_profiles.append(s.getvalue())
    except ImportError:
        print("Rust extension not available")
        rs_times = [np.nan]
        rs_profiles = ["N/A"]

    # Print detailed statistics
    print("\nPython Implementation Performance:")
    print(f"Average time: {np.mean(py_times):.6f} ± {np.std(py_times):.6f} seconds")

    print("\nRust Implementation Performance:")
    print(f"Average time: {np.mean(rs_times):.6f} ± {np.std(rs_times):.6f} seconds")

    # Print first profile for detailed inspection
    print("\nPython Profiling (First Run):")
    print(py_profiles[0])
    print("\nRust Profiling (First Run):")
    print(rs_profiles[0])

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.boxplot([py_times, rs_times], labels=["Python", "Rust"])
    plt.title(f"Whitening Performance Comparison (Matrix Size: {size}x{size})")
    plt.ylabel("Computation Time (seconds)")
    plt.tight_layout()
    plt.show()


def detailed_step_breakdown(size=1000, repeats=10):
    """
    Detailed step-by-step performance breakdown
    """
    np.random.seed(42)
    data = np.random.randn(size, size)

    # Python implementation steps
    def python_steps():
        start = time.time()
        cov = data.T @ data / len(data)
        cov_time = time.time() - start

        start = time.time()
        U, S, Ut = np.linalg.svd(cov, full_matrices=True)
        svd_time = time.time() - start

        start = time.time()
        W = (U @ np.diag(1 / np.sqrt(S + 1e-6))) @ Ut
        whitening_time = time.time() - start

        return {
            "covariance_computation": cov_time,
            "svd_computation": svd_time,
            "whitening_matrix_computation": whitening_time,
            "total": cov_time + svd_time + whitening_time,
        }

    # Rust implementation steps
    def rust_steps():
        try:
            from dspant._rs import compute_whitening_matrix

            start = time.time()
            cov = (data.T @ data / len(data)).astype(np.float32)
            cov_time = time.time() - start

            start = time.time()
            W = compute_whitening_matrix(cov, 1e-6)
            total_time = time.time() - start

            return {
                "covariance_computation": cov_time,
                "whitening_computation": total_time,
                "total": total_time + cov_time,
            }
        except ImportError:
            print("Rust extension not available")
            return {}

    # Run multiple times and aggregate results
    print("\nPython Method Step-by-Step Breakdown:")
    py_results = [python_steps() for _ in range(repeats)]
    py_summary = {
        k: {
            "mean": np.mean([r[k] for r in py_results]),
            "std": np.std([r[k] for r in py_results]),
        }
        for k in py_results[0].keys()
    }

    for step, stats in py_summary.items():
        print(f"{step}: {stats['mean']:.6f} ± {stats['std']:.6f} seconds")

    print("\nRust Method Performance:")
    rs_results = [rust_steps() for _ in range(repeats)]
    if rs_results and rs_results[0]:  # Check if results are not empty
        rs_summary = {
            k: {
                "mean": np.mean([r[k] for r in rs_results]),
                "std": np.std([r[k] for r in rs_results]),
            }
            for k in rs_results[0].keys()
        }

        for step, stats in rs_summary.items():
            print(f"{step}: {stats['mean']:.6f} ± {stats['std']:.6f} seconds")


# %%
# Run the performance analysis
detailed_performance_comparison()
# %%
detailed_step_breakdown()
