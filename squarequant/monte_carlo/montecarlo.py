# squarequant/monte_carlo/montecarlo_test.py
import numpy as np
from numpy.random import default_rng, Generator, PCG64, SeedSequence
from scipy import stats
from scipy.linalg import cholesky, LinAlgError
import pandas as pd
from squarequant.constants import DEFAULT_MC_CONFIDENCE, TRADING_DAYS_PER_YEAR

class MonteCarloSimulator:
    """
    A class for performing Monte Carlo simulations and related statistical analyses.
    """
    def __init__(self, confidence_level=DEFAULT_MC_CONFIDENCE):
        """
        Initialize the MonteCarloSimulator with a confidence level.
        Parameters:
        -----------
        confidence_level : float, optional (default=DEFAULT_MC_CONFIDENCE)
            The confidence level for confidence intervals (e.g., 0.95 for 95% CI).
        """
        self.confidence_level = confidence_level

    def simple_mc(self, sample, distribution: str = 'normal', df: int = 4) -> dict:
        """Computes sample statistics and confidence intervals for the mean under specified distributions.

        Supports normal, Student's t, and Laplace distributions for modeling data with varying tail behavior.
        The function is designed for robust statistical inference, especially in financial or heavy-tailed contexts.

        Args:
            sample (np.ndarray: Input data sample. Must be 1D and non-empty.
                Expected to contain numeric values (int/float).
            distribution (str, optional): Distribution used for confidence interval calculation.
                Defaults to 'normal'. Options:
                - 'normal': Uses standard normal (Gaussian) distribution.
                - 't': Uses Student's t-distribution (adjustable tail heaviness via `df`).
                - 'laplace': Uses Laplace distribution (sharp peak, fat tails).
            df (int, optional): Degrees of freedom for Student's t-distribution. Defaults to 4.
                For 't' distribution, must be > 0. Values 3-5 are typical for financial data;
                values >30 approximate normal behavior.

        Returns:
            dict[str, float | tuple[float, float]]:
                A dictionary with keys:
                - 'sample_mean': Sample mean (float).
                - 'unbiased_variance': Unbiased sample variance (float, Bessel's correction).
                - 'confidence_interval': Tuple of (lower, upper) bounds for the mean (float, float).
                - '_metadata': Debug metadata (dict), including distribution, df (if applicable), and sample size.

        Raises:
            ValueError: If `sample` is empty, `distribution` is unsupported, or `df` <= 0 for 't'.
            TypeError: If `sample` is not array-like or contains non-numeric values.
            RuntimeError: If numerical errors occur during critical value calculation.

        Notes:
            For 't' distribution, `df` must be > 0. For `df` <= 2, variance is theoretically infinite.
            Laplace distribution assumes location=0, scale=1 (standardized shocks).
            Confidence intervals are symmetric; adjust alpha for one-sided tests.
            The confidence level is set by the class attribute confidence_level`.
        """
        # --- Input validation ---
        if not isinstance(sample, (np.ndarray, list)):
            raise TypeError("`sample` must be array-like.")
        if len(sample) == 0:
            raise ValueError("`sample` cannot be empty.")
        if distribution == 't' and df <= 0:
            raise ValueError("`df` must be > 0 for Student's t-distribution.")

        # --- Core logic ---
        n = len(sample)
        sample_mean = float(np.mean(sample))  # Explicit float for JSON serialization
        unbiased_variance = np.var(sample, ddof=1)  # ddof=1 for unbiased estimator
        alpha = 1.0 - self.confidence_level
        std_error = np.sqrt(unbiased_variance / n)

        # --- Distribution-specific critical values ---
        try:
            if distribution == 'normal':
                z_critical = stats.norm.ppf(1.0 - alpha / 2)
            elif distribution == 't':
                z_critical = stats.t.ppf(1.0 - alpha / 2, df=df)
            elif distribution == 'laplace':
                z_critical = stats.laplace.ppf(1.0 - alpha / 2)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}. Use 'normal', 't', or 'laplace'.")
        except LinAlgError as e:
            raise RuntimeError(f"Numerical error in {distribution} PPF: {e}")

        margin_of_error = z_critical * std_error
        ci = (sample_mean - margin_of_error, sample_mean + margin_of_error)

        return {
            'sample_mean': sample_mean,
            'unbiased_variance': unbiased_variance,
            'confidence_interval': ci,
            '_metadata': {  # Debug metadata
                'distribution': distribution,
                'df': df if distribution == 't' else None,
                'sample_size': n,
            }
        }

    @staticmethod
    def simple_mc_static(
            sample: np.ndarray | list,
            confidence_level: float = DEFAULT_MC_CONFIDENCE,
            distribution: str = 'normal',
            df: int = 4
    ) -> dict[str, float | tuple[float, float]]:
        """Computes sample statistics and confidence intervals for the mean under specified distributions.

        Supports normal, Student's t, and Laplace distributions for modeling data with varying tail behavior.
        The function is designed for robust statistical inference, especially in financial or heavy-tailed contexts.

        Args:
            sample (np.ndarray: Input data sample. Must be 1D and non-empty.
                Expected to contain numeric values (int/float).
            distribution (str, optional): Distribution used for confidence interval calculation.
                Defaults to 'normal'. Options:
                - 'normal': Uses standard normal (Gaussian) distribution.
                - 't': Uses Student's t-distribution (adjustable tail heaviness via `df`).
                - 'laplace': Uses Laplace distribution (sharp peak, fat tails).
            df (int, optional): Degrees of freedom for Student's t-distribution. Defaults to 4.
                For 't' distribution, must be > 0. Values 3-5 are typical for financial data;
                values >30 approximate normal behavior.

        Returns:
            dict[str, float | tuple[float, float]]:
                A dictionary with keys:
                - 'sample_mean': Sample mean (float).
                - 'unbiased_variance': Unbiased sample variance (float, Bessel's correction).
                - 'confidence_interval': Tuple of (lower, upper) bounds for the mean (float, float).
                - '_metadata': Debug metadata (dict), including distribution, df (if applicable), and sample size.

        Raises:
            ValueError: If `sample` is empty, `distribution` is unsupported, or `df` <= 0 for 't'.
            TypeError: If `sample` is not array-like or contains non-numeric values.
            RuntimeError: If numerical errors occur during critical value calculation.

        Notes:
            For 't' distribution, `df` must be > 0. For `df` <= 2, variance is theoretically infinite.
            Laplace distribution assumes location=0, scale=1 (standardized shocks).
            Confidence intervals are symmetric; adjust alpha for one-sided tests.
            The confidence level is set by the class attribute`.
        """
        # --- Input validation ---
        if not isinstance(sample, (np.ndarray, list)):
            raise TypeError("`sample` must be array-like.")
        if len(sample) == 0:
            raise ValueError("`sample` cannot be empty.")
        if not 0 < confidence_level < 1:
            raise ValueError("`confidence_level` must be in (0, 1).")
        if distribution == 't' and df <= 0:
            raise ValueError("`df` must be > 0 for Student's t-distribution.")

        # --- Core logic ---
        n = len(sample)
        sample_mean = float(np.mean(sample))
        unbiased_variance = np.var(sample, ddof=1)
        alpha = 1.0 - confidence_level
        std_error = np.sqrt(unbiased_variance / n)

        try:
            if distribution == 'normal':
                z_critical = stats.norm.ppf(1.0 - alpha / 2)
            elif distribution == 't':
                z_critical = stats.t.ppf(1.0 - alpha / 2, df=df)
            elif distribution == 'laplace':
                z_critical = stats.laplace.ppf(1.0 - alpha / 2)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}.")
        except Exception as e:
            raise RuntimeError(f"Error calculating critical value: {e}")

        margin_of_error = z_critical * std_error
        ci = (sample_mean - margin_of_error, sample_mean + margin_of_error)

        return {
            'sample_mean': sample_mean,
            'unbiased_variance': unbiased_variance,
            'confidence_interval': ci,
            '_metadata': {
                'distribution': distribution,
                'df': df if distribution == 't' else None,
                'sample_size': n,
                'confidence_level': confidence_level,
            }
        }

    def brownian_paths(self, T=1.0, N=10000, nsims=500, seed=None, dtype=np.float32) -> np.ndarray:
        """Simulates multiple Brownian motion paths using exact discretization.

        Generates `nsims` independent Brownian motion trajectories over the interval [0, T]
        with `N` discrete time points. The simulation is optimized for performance and supports
        reproducibility via a random seed.

        Args:
            T (float, optional): Time horizon of the simulation. Defaults to 1.0.
            N (int, optional): Number of discrete time points in each path. Defaults to 10000.
            nsims (int, optional): Number of Brownian motion trajectories to simulate. Defaults to 500.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            dtype (numpy.dtype, optional): Data type for the output array. Defaults to np.float32.
                Supported types: np.float32, np.float64.

        Returns:
            numpy.ndarray:
                A 2D array of shape (nsims, N) containing `nsims` Brownian motion paths.
                Each row represents a single trajectory.

        Note:
            The simulation uses the exact method: increments are drawn from a normal distribution
            with mean 0 and variance dt, where dt = T/(N-1).
        """
        rng = default_rng(seed)
        dt = T / (N - 1)
        W = np.zeros((nsims, N), dtype=dtype)
        W[:, 0] = 0.0
        increments = rng.normal(0.0, np.sqrt(dt), (nsims, N - 1)).astype(dtype)
        np.cumsum(increments, axis=1, out=W[:, 1:])
        return W

    @staticmethod
    def brownian_paths_static(T=1.0, N=10000, nsims=500, seed=None, dtype=np.float32):
        """Simulates multiple Brownian motion paths using exact discretization.
           Generates `nsims` independent Brownian motion trajectories over the interval [0, T]
           with `N` discrete time points. The simulation is optimized for performance and supports
           reproducibility via a random seed.

       Args:
           T (float, optional): Time horizon of the simulation. Defaults to 1.0.
           N (int, optional): Number of discrete time points in each path. Defaults to 10000.
           nsims (int, optional): Number of Brownian motion trajectories to simulate. Defaults to 500.
           seed (int, optional): Random seed for reproducibility. Defaults to None.
           dtype (numpy.dtype, optional): Data type for the output array. Defaults to np.float32.
               Supported types: np.float32, np.float64.

       Returns:
           numpy.ndarray:
           A 2D array of shape (nsims, N) containing `nsims` Brownian motion paths.
           Each row represents a single trajectory.

       Note:
           The simulation uses the exact method: increments are drawn from a normal distribution
           with mean 0 and variance dt, where dt = T/(N-1).
       """

        rng = default_rng(seed)
        dt = T / (N - 1)
        W = np.zeros((nsims, N), dtype=dtype)
        W[:, 0] = 0.0
        increments = rng.normal(0.0, np.sqrt(dt), (nsims, N - 1)).astype(dtype)
        np.cumsum(increments, axis=1, out=W[:, 1:])
        return W

    def correlated_brownian_paths(
            self,
            T: float = 1.0,
            N: int = 10000,
            nsims: int = 500,
            correlation_matrix: np.ndarray = None,
            seed: int = None,
            dtype: type = np.float32,
    ) -> np.ndarray:
        """Simulates M correlated Brownian motion paths using Cholesky decomposition and vectorized operations.
            Generates `nsims` trajectories of M correlated Brownian motions over the interval [0, T],
            discretized into `N` points. Each simulation uses a unique random stream derived from a master seed,
            ensuring reproducibility and independence across simulations.

        Args:
            T (float, optional): Time horizon of the simulation. Defaults to 1.0.
            N (int, optional): Number of discrete time points in each path. Defaults to 10000.
            nsims (int, optional): Number of trajectories to simulate. Defaults to 500.
            correlation_matrix (numpy.ndarray, optional): M x M correlation matrix. If None, an identity matrix is used (independent paths). Defaults to None.
            seed (int, optional): Master random seed for reproducibility. If None, the global random state is used. Defaults to None.
            dtype (type, optional): Data type for the output array. Defaults to np.float32.

        Returns:
            np.ndarray:
                A 3D array of shape (M, nsims, N) containing M correlated Brownian motion paths.
                The first dimension corresponds to the assets, the second to the simulations, and the third to time.

        Raises:
            ValueError: If the correlation matrix is not square, symmetric, or positive definite.
        """
        if correlation_matrix is None:
            M = 1
            correlation_matrix = np.eye(M, dtype=dtype)
        else:
            correlation_matrix = np.asarray(correlation_matrix, dtype=dtype)
            M = correlation_matrix.shape[0]
            if correlation_matrix.shape[1] != M:
                raise ValueError("Correlation matrix must be square.")
            if not np.allclose(correlation_matrix, correlation_matrix.T):
                raise ValueError("Correlation matrix must be symmetric.")
            try:
                L = cholesky(correlation_matrix, lower=True)
            except LinAlgError:
                raise ValueError("Correlation matrix must be positive definite.")

        master_seed = SeedSequence(seed)
        child_seeds = master_seed.spawn(nsims)
        rngs = [Generator(PCG64(s)) for s in child_seeds]
        dt = T / (N - 1)
        Z = np.stack([rng.standard_normal((N - 1, M)) for rng in rngs]).astype(dtype) * np.sqrt(dt)
        if M > 1:
            Z = np.einsum('ij,...j->...i', L, Z)
        W = np.zeros((M, nsims, N), dtype=dtype)
        W[:, :, 1:] = np.transpose(np.cumsum(Z, axis=1), (2, 0, 1))
        return W

    @staticmethod
    def correlated_brownian_paths_static(
            T: float = 1.0,
            N: int = 10000,
            nsims: int = 500,
            correlation_matrix: np.ndarray = None,
            seed: int = None,
            dtype: type = np.float32,
    ) -> np.ndarray:
        """Simulates M correlated Brownian motion paths using Cholesky decomposition and vectorized operations.
            Generates `nsims` trajectories of M correlated Brownian motions over the interval [0, T],
            discretized into `N` points. Each simulation uses a unique random stream derived from a master seed,
            ensuring reproducibility and independence across simulations.

        Args:
            T (float, optional): Time horizon of the simulation. Defaults to 1.0.
            N (int, optional): Number of discrete time points in each path. Defaults to 10000.
            nsims (int, optional): Number of trajectories to simulate. Defaults to 500.
            correlation_matrix (numpy.ndarray, optional): M x M correlation matrix. If None, an identity matrix is used (independent paths). Defaults to None.
            seed (int, optional): Master random seed for reproducibility. If None, the global random state is used. Defaults to None.
            dtype (type, optional): Data type for the output array. Defaults to np.float32.

        Returns:
            np.ndarray:
                A 3D array of shape (M, nsims, N) containing M correlated Brownian motion paths.
                The first dimension corresponds to the assets, the second to the simulations, and the third to time.

        Raises:
            ValueError: If the correlation matrix is not square, symmetric, or positive definite.
        """
        if correlation_matrix is None:
            M = 1
            correlation_matrix = np.eye(M, dtype=dtype)
        else:
            correlation_matrix = np.asarray(correlation_matrix, dtype=dtype)
            M = correlation_matrix.shape[0]
            if correlation_matrix.shape[1] != M:
                raise ValueError("Correlation matrix must be square.")
            if not np.allclose(correlation_matrix, correlation_matrix.T):
                raise ValueError("Correlation matrix must be symmetric.")
            try:
                L = cholesky(correlation_matrix, lower=True)
            except LinAlgError:
                raise ValueError("Correlation matrix must be positive definite.")

        master_seed = SeedSequence(seed)
        child_seeds = master_seed.spawn(nsims)
        rngs = [Generator(PCG64(s)) for s in child_seeds]
        dt = T / (N - 1)
        Z = np.stack([rng.standard_normal((N - 1, M)) for rng in rngs]).astype(dtype) * np.sqrt(dt)
        if M > 1:
            Z = np.einsum('ij,...j->...i', L, Z)
        W = np.zeros((M, nsims, N), dtype=dtype)
        W[:, :, 1:] = np.transpose(np.cumsum(Z, axis=1), (2, 0, 1))
        return W

    def gbm_paths(
            self,
            S0: float = 1.0,
            mu: float = 0.1,
            sigma: float = 0.2,
            T: float = 1.0,
            N: int = 10000,
            nsims: int = 500,
            seed: int = None,
            dtype: type = np.float32
    ) -> np.ndarray:
        """Simulates Geometric Brownian Motion (GBM) paths for asset price modeling.

        Generates `nsims` trajectories of an asset price following the GBM process:
        dS_t = mu * S_t * dt + sigma * S_t * dW_t,
        where S_t is the asset price, mu is the drift, sigma is the volatility, and W_t is a Brownian motion.

        Args:
            S0 (float, optional): Initial asset price. Defaults to 1.0.
            mu (float, optional): Drift (expected return). Defaults to 0.1.
            sigma (float, optional): Volatility. Defaults to 0.2.
            T (float, optional): Time horizon of the simulation. Defaults to 1.0.
            N (int, optional): Number of discrete time points in each path. Defaults to 10000.
            nsims (int, optional): Number of trajectories to simulate. Defaults to 500.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            dtype (type, optional): Data type for the output array. Defaults to np.float32.

        Returns:
            np.ndarray:
                A 2D array of shape (nsims, N) containing `nsims` GBM paths.
                Each row represents a single price trajectory.
        """
        W = self.brownian_paths(T, N, nsims, seed, dtype)
        time_grid = np.linspace(0, T, N, dtype=dtype)
        drift = (mu - 0.5 * sigma ** 2) * time_grid
        diffusion = sigma * W
        S = S0 * np.exp(drift + diffusion)
        return S

    def gbm_correlated_paths(
            self,
            S0: np.ndarray,  # Initial prices (shape: M,)
            mu: np.ndarray,  # Drifts (shape: M,)
            sigma: np.ndarray,  # Volatilities (shape: M,)
            T: float = 1.0,
            N: int = 10000,
            nsims: int = 500,
            correlation_matrix: np.ndarray = None,
            seed: int = None,
            dtype: type = np.float32
    ) -> np.ndarray:
        """
        Simulate correlated Geometric Brownian Motion paths for M assets (fully vectorized).

        Parameters:
        -----------
        S0 : np.ndarray (shape: M,)
            Initial asset prices.
        mu : np.ndarray (shape: M,)
            Drift for each asset.
        sigma : np.ndarray (shape: M,)
            Volatility for each asset.
        T, N, nsims, correlation_matrix, seed, dtype : as before.

        Returns:
        --------
        np.ndarray (shape: M x nsims x N)
            Correlated GBM paths for M assets.
        """
        M = len(S0)
        W = self.correlated_brownian_paths(T, N, nsims, correlation_matrix, seed, dtype)
        time_grid = np.linspace(0, T, N, dtype=dtype)

        # Reshape for broadcasting: (M, 1, N) and (M, nsims, N)
        drift = (mu - 0.5 * sigma ** 2)[:, np.newaxis, np.newaxis] * time_grid[np.newaxis, np.newaxis, :]
        diffusion = sigma[:, np.newaxis, np.newaxis] * W

        S = S0[:, np.newaxis, np.newaxis] * np.exp(drift + diffusion)
        return S

    @staticmethod
    def gbm_paths_static(
        S0: float = 1.0,
        mu: float = 0.1,
        sigma: float = 0.2,
        T: float = 1.0,
        N: int = 10000,
        nsims: int = 500,
        seed: int = None,
        dtype: type = np.float32
    ) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion paths.

        Parameters:
        -----------
        S0 : float, optional (default=1.0)
            Initial asset price.
        mu : float, optional (default=0.1)
            Drift (expected return).
        sigma : float, optional (default=0.2)
            Volatility.
        T : float, optional (default=1.0)
            Time horizon.
        N : int, optional (default=10000)
            Number of time steps.
        nsims : int, optional (default=500)
            Number of simulations.
        seed : int, optional (default=None)
            Random seed for reproducibility.
        dtype : type, optional (default=np.float32)
            Data type for the output array.

        Returns:
        --------
        numpy.ndarray
            Array of shape (nsims, N) representing nsims GBM paths.
        """
        simulator = MonteCarloSimulator()
        W = simulator.brownian_paths(T, N, nsims, seed, dtype)
        time_grid = np.linspace(0, T, N, dtype=dtype)
        drift = (mu - 0.5 * sigma ** 2) * time_grid
        diffusion = sigma * W
        S = S0 * np.exp(drift + diffusion)
        return S

    @staticmethod
    def gbm_correlated_paths_static(
            S0: np.ndarray,  # Initial prices (shape: M,)
            mu: np.ndarray,  # Drifts (shape: M,)
            sigma: np.ndarray,  # Volatilities (shape: M,)
            T: float = 1.0,
            N: int = 10000,
            nsims: int = 500,
            correlation_matrix: np.ndarray = None,
            seed: int = None,
            dtype: type = np.float32
    ) -> np.ndarray:
        """Simulates correlated Geometric Brownian Motion (GBM) paths for M assets.

        Generates `nsims` trajectories for M assets, each following a GBM process with specified
        initial prices, drifts, volatilities, and correlation structure. The simulation is fully vectorized
        for performance.

        Args:
            S0 (np.ndarray): Initial asset prices, shape (M,).
            mu (np.ndarray): Drift for each asset, shape (M,).
            sigma (np.ndarray): Volatility for each asset, shape (M,).
            T (float, optional): Time horizon of the simulation. Defaults to 1.0.
            N (int, optional): Number of discrete time points in each path. Defaults to 10000.
            nsims (int, optional): Number of trajectories to simulate. Defaults to 500.
            correlation_matrix (np.ndarray, optional): M x M correlation matrix. If None, an identity matrix is used. Defaults to None.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            dtype (type, optional): Data type for the output array. Defaults to np.float32.

        Returns:
            np.ndarray:
                A 3D array of shape (M, nsims, N) containing correlated GBM paths for M assets.
                The first dimension corresponds to the assets, the second to the simulations, and the third to time.
        """
        M = len(S0)
        simulator = MonteCarloSimulator()
        W = simulator.correlated_brownian_paths(T, N, nsims, correlation_matrix, seed, dtype)
        time_grid = np.linspace(0, T, N, dtype=dtype)

        # Reshape for broadcasting: (M, 1, N) and (M, nsims, N)
        drift = (mu - 0.5 * sigma ** 2)[:, np.newaxis, np.newaxis] * time_grid[np.newaxis, np.newaxis, :]
        diffusion = sigma[:, np.newaxis, np.newaxis] * W

        S = S0[:, np.newaxis, np.newaxis] * np.exp(drift + diffusion)
        return S

    def fit_params(
            self,
            data: pd.DataFrame,
            model: str = 'gbm',
            dt: float = 1 / TRADING_DAYS_PER_YEAR
    ) -> tuple[float, float]:
        """Estimates the drift and volatility parameters from historical price data.

        Fits the specified model to the provided historical price data to estimate the drift (mu)
        and volatility (sigma). Currently, only the Geometric Brownian Motion (GBM) model is supported.

        Args:
            data (pd.DataFrame): DataFrame containing historical price data (e.g., closing prices).
            model (str, optional): Model to fit. Currently only 'gbm' is supported. Defaults to 'gbm'.
            dt (float, optional): Time interval between data points (e.g., 1/252 for daily data). Defaults to 1/TRADING_DAYS_PER_YEAR.

        Returns:
            tuple[float, float]:
                Estimated drift (mu) and volatility (sigma).

        Raises:
            ValueError: If the specified model is not supported (at this stage only supports gbm).
        """
        if model.lower() == 'gbm':
            prices = data.values.flatten()
            log_returns = np.log(prices[1:] / prices[:-1])
            mu = np.mean(log_returns) / dt
            sigma = np.std(log_returns) / np.sqrt(dt)
            return mu, sigma
        else:
            raise ValueError(f"Model '{model}' not supported. Currently only 'gbm' is supported.")