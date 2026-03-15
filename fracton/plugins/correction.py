"""CorrectionPlugin — applies force-specific PAC correction templates to results."""

from __future__ import annotations

from typing import Any, Optional

from fracton.corrections.template import CorrectionTemplate, correction


class CorrectionPlugin:
    """Apply a PAC correction template at crystallization.

    Uses the universal template 1 ± F_a/(nπF_b²) to adjust results
    by the appropriate force correction factor.

    Args:
        template: A CorrectionTemplate to apply. If None, uses the
            EM correction (a=3, b=4, n=1, sign=+1) as default.
    """

    def __init__(self, template: Optional[CorrectionTemplate] = None):
        if template is None:
            # Default: EM correction
            template = correction(a=3, b=4, n=1, sign=1)
        self._template = template

    @property
    def name(self) -> str:
        return "correction"

    def on_recurse(self, context: Any, depth: int) -> None:
        """No-op — corrections are applied at crystallization."""
        pass

    def on_crystallize(self, context: Any, result: Any) -> Any:
        """Apply the correction factor to numeric results."""
        if isinstance(result, (int, float)):
            return result * self._template.factor
        return result

    def validate(self, context: Any) -> bool:
        """Always valid — corrections don't gate recursion."""
        return True
