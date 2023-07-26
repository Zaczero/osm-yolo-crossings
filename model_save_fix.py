from keras.models import Model


def model_save_fix(model: Model) -> None:
    save_ = model.save

    def save(*args, **kwargs):
        kwargs.pop('options', None)
        return save_(*args, **kwargs)

    model.save = save
