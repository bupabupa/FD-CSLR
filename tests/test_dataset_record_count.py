from dataset.splits import indexed_record_count


def test_metadata_does_not_reduce_record_count():
    records = {
        "prefix": "/dataset",
        0: {"fileid": "a"},
        1: {"fileid": "b"},
        2: {"fileid": "c"},
    }
    assert indexed_record_count(records) == 3
    assert indexed_record_count([{"fileid": "a"}, {"fileid": "b"}]) == 2


def test_sparse_record_keys_are_rejected():
    try:
        indexed_record_count({"prefix": "/dataset", 0: {}, 2: {}})
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("sparse record keys were accepted")


if __name__ == "__main__":
    test_metadata_does_not_reduce_record_count()
    test_sparse_record_keys_are_rejected()
    print("dataset record count tests passed")
