import wrist2vec
import wrist2vec.adapt as adapt_module
import wrist2vec.finetune as finetune_module
import wrist2vec.infer as infer_module
import wrist2vec.pretrain as pretrain_module
from wrist2vec.downstream_model import Wrist2vecDownstreamModel
from wrist2vec.pretrain_model import Wrist2vecPretrainModel
from wrist2vec.wrist2vec_adaptation import Wrist2vecAdaptation
from wrist2vec.wrist2vec_finetuning import Wrist2vecFinetuning
from wrist2vec.wrist2vec_modelling import Wrist2vecPretraining


def test_wrist2vec_package_and_entrypoints_import():
    assert wrist2vec.__all__ == []
    assert callable(pretrain_module.wrist2vec_pretrain)
    assert callable(adapt_module.wrist2vec_adapt)
    assert callable(finetune_module.supervised)
    assert callable(infer_module.run_inference)


def test_wrist2vec_public_classes_resolve():
    assert Wrist2vecPretrainModel.__name__ == "Wrist2vecPretrainModel"
    assert Wrist2vecDownstreamModel.__name__ == "Wrist2vecDownstreamModel"
    assert Wrist2vecPretraining.__name__ == "Wrist2vecPretraining"
    assert Wrist2vecAdaptation.__name__ == "Wrist2vecAdaptation"
    assert Wrist2vecFinetuning.__name__ == "Wrist2vecFinetuning"
