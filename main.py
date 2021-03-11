from DogsCats import DogsCatsClassificator

if __name__ == "__main__":
    classificator = DogsCatsClassificator()

    model = classificator.create_model()
    # classificator.train(model, "modelv09.h5")

    # for testing
    model.load_weights("modelv8.h5")
    classificator.manualTest(model)