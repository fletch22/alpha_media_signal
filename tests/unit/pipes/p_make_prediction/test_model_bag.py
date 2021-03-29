from ams.pipes.p_make_prediction.ModelBag import TrainingBag


class Foo():
    pass


def test_model_bag():
    # Arrange
    model_bag = TrainingBag()
    model = Foo()

    # Act
    model_bag.add_fistfull(rev_ndx=0, purchase_date_str="2020-01-01", model=model)

    # Assert
    assert (len(model_bag.models) == 1)