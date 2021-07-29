from typing import ClassVar, List, Optional, Type

import pytest
from sequoia.common.config import Config
from sequoia.methods.method_test import MethodTests
from sequoia.settings import Setting
from sequoia.settings.sl import ContinualSLSetting, TaskIncrementalSLSetting

from real_deel_dark_experience import DER

class TestDERMethod(MethodTests):
    """ Tests for DER Method.
    
    The main test of interest is `test_debug`, which is implemented in the MethodTests
    class.
    """
    
    Method: ClassVar[Type[DER]] = DER

    @pytest.mark.skip(reason="A bit too long to run")
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize(
        "setting_type",
        [
            ContinualSLSetting,
            TaskIncrementalSLSetting,
        ]
    )
    def test_cndpm_method_on_settings(
            self,
            setting_type: Type[Setting]
    ):
        setting = setting_type(dataset="mnist", nb_tasks=5) 
        hparams = DER.HParams()
        method = self.Method(hparams)

        results = setting.apply(method)
        print(results.summary())

    @classmethod
    @pytest.fixture
    def method(cls, config: Config) -> DER:
        """ Fixture that returns the Method instance to use when testing/debugging.
        Needs to be implemented when creating a new test class (to generate tests for a
        new method).
        """
        debug_hparams = DER.HParams(device=config.device)
        if config.debug:
            # TODO: Set the parameters for a debugging run on a short setting!
            # (Need runs to be shorter than 30 secs per setting!)
            debug_hparams.dpmoe = DER(
                buffer_size=25,
                max_epochs_per_task=1,
            )       

        return cls.Method(debug_hparams)

    def validate_results(self,
                         setting: Setting,
                         method: DER,
                         results: Setting.Results,
                         ) -> None:
        assert results
        assert results.objective