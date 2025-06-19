from mediapipe_model_maker import gesture_recognizer


HPARAMS = gesture_recognizer.HParams(
    export_dir="./training/exported_model",
    batch_size=32,
    epochs=100,
    learning_rate=0.0005,
    lr_decay=0.995
)
OPTIONS = gesture_recognizer.GestureRecognizerOptions(hparams=HPARAMS)


def main():
    data = gesture_recognizer.Dataset.from_folder(
        dirname="./training/gestures-dataset",
        hparams=gesture_recognizer.HandDataPreprocessingParams(
            shuffle=True
        )
    )
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)

    
    model = gesture_recognizer.GestureRecognizer.create(
        train_data=train_data,
        validation_data=validation_data,
        options=OPTIONS
    )

    loss, acc = model.evaluate(test_data, batch_size=1)
    print(f"Test loss:{loss}, Test accuracy:{acc}")

    model.export_model()


if __name__ == "__main__":
    main()
