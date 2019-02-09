from kiwi.models.linear.linear_trainer import LinearTrainer


class LinearWordQETrainer(LinearTrainer):
    def __init__(
        self, model, optimizer_name, regularization_constant, checkpointer
    ):
        super().__init__(
            classifier=model,
            checkpointer=checkpointer,
            algorithm=optimizer_name,
            regularization_constant=regularization_constant,
        )

    @property
    def model(self):
        return self.classifier
