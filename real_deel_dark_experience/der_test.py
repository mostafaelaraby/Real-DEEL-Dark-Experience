from typing import ClassVar, List, Optional, Type

import pytest
from sequoia.common.config import Config
from sequoia.methods.method_test import MethodTests
from sequoia.settings import Setting
from sequoia.settings.sl import ContinualSLSetting, TaskIncrementalSLSetting, SLSetting

from real_deel_dark_experience import DER


class TestDERMethod(MethodTests):
    """Tests for DER Method.

    The main test of interest is `test_debug`, which is implemented in the MethodTests
    class.
    """

    Method: ClassVar[Type[DER]] = DER

    @classmethod
    @pytest.fixture
    def method(cls, session_config: Config, setting_type: Type[Setting]) -> DER:
        """Fixture that returns the Method instance to use when testing/debugging.
        Needs to be implemented when creating a new test class (to generate tests for a
        new method).
        """
        # NOTE: DER method doesn't yet work with batched rl environments.
        batch_size = 32 if issubclass(setting_type, SLSetting) else None
        hparams = DER.HParams(batch_size=batch_size)
        if session_config.debug:
            # TODO: Set the parameters for a debugging run on a short setting!
            # (Need runs to be shorter than 30 secs per setting!)
            hparams = DER.HParams(
                buffer_size=100,
                replay_minibatch_size=25,
                max_epochs_per_task=1,
                batch_size=batch_size,
            )
        return cls.Method(hparams=hparams)

    def validate_results(
        self,
        setting: Setting,
        method: DER,
        results: Setting.Results,
    ) -> None:
        assert results
        assert results.objective
        # TODO: Add more rigorous testing, checking that the performance makes sense for
        # the given setting, dataset, method, etc.
        # See (https://github.com/lebrice/Sequoia/blob/master/examples/basic/pl_example_test.py)
        # for an example.
