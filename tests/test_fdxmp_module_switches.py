from fd_xmp_network import FDXMPModel


BASE = {
    "num_classes": 4,
    "c2d_type": "resnet18",
    "conv_type": 2,
    "use_bn": 1,
}


def _expect_requirement_error(loss_weights, **kwargs):
    try:
        FDXMPModel(
            **BASE,
            gloss_dict={"a": (1,), "b": (2,), "c": (3,)},
            loss_weights=loss_weights,
            **kwargs
        )
    except ValueError as error:
        assert "requires" in str(error)
    else:
        raise AssertionError("inactive module loss was accepted")


def test_loss_keys_require_their_modules():
    _expect_requirement_error({"SeqCTC": 1.0, "WeightedCTC": 0.5})
    _expect_requirement_error({"SeqCTC": 1.0, "PRA": 0.05})
    _expect_requirement_error({"SeqCTC": 1.0, "RDBCTC": 0.05})
    _expect_requirement_error({"SeqCTC": 1.0, "ProtoAlign": 0.01}, proto_enabled=False)
    _expect_requirement_error(
        {"SeqCTC": 1.0, "TextAlign": 0.01},
        text_proto_enabled=False,
    )
    _expect_requirement_error(
        {"SeqCTC": 1.0, "SignerCE": 0.01},
        signer_film=False,
    )
    _expect_requirement_error(
        {"SeqCTC": 1.0, "RDBStyle": 0.02},
        signer_film=False,
    )


def test_rdb_requires_visual_prototypes():
    _expect_requirement_error(
        {"SeqCTC": 1.0, "RDBRecon": 0.05},
        proto_enabled=False,
        rdb_enabled=True,
    )


def test_enabled_rdb_requires_an_active_loss():
    _expect_requirement_error(
        {"SeqCTC": 1.0},
        rdb_enabled=True,
    )
    _expect_requirement_error(
        {"SeqCTC": 1.0, "RDBRecon": 0.05},
        rdb_enabled=True,
    )


if __name__ == "__main__":
    test_loss_keys_require_their_modules()
    test_rdb_requires_visual_prototypes()
    test_enabled_rdb_requires_an_active_loss()
    print("FD-XMP module switch tests passed")
