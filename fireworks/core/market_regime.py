"""Market regime definitions and utilities."""


class MarketRegime:
    """Represents a market regime with associated parameters."""

    def __init__(self, name, mean_return, volatility, correlation=None):
        """
        Initialize a market regime.

        Args:
            name: Name of the regime
            mean_return: Expected annual return
            volatility: Annual volatility
            correlation: Asset correlations (optional)
        """
        self.name = name
        self.mean_return = mean_return
        self.volatility = volatility
        self.correlation = correlation
