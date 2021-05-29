from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import os


class Trainer:
    """
    Trains model using GPT2

    Attributes:
        model: transformers.GPT2LMHeadModel
            GPT2 Model
        device: torch.device
            Cuda GPU to train on
        train_dataloader: torch.utils.data.DataLoader
            Dataloader to train model upon
        val_dataloader: torch.utils.data.DataLoader or None
            Dataloader to validate model upon
    """

    def __init__(self, train_dataloader, val_dataloader = None):
        """
        Initialises Trainer by defining model and GPU

        Args:
        train_dataloader: torch.utils.data.DataLoader
            Dataloader to train model upon, obtained from Dataloader class
        val_dataloader: Optional torch.utils.data.DataLoader
            Dataloader to validate model upon obtained from DataLoader class,
            not required if Trainer is only used for final training
        """

        # Create GPT2 Config
        config = GPT2Config.from_pretrained("gpt2")

        # Load language head model and input default config
        model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

        # Recreate tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                                  pad_token='<|pad|>')

        # Tell model we have added bos, eos, pad token
        model.resize_token_embeddings(len(tokenizer))

        # Tell pytorch to run this model on the GPU.
        device = torch.device("cuda")
        model.cuda()

        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def validate(self, epochs, learning_rate, warmup_steps):
        """
        Validates model, prints train and validation loss on each epoch

        Args:
            epochs: int
            learning_rate: float
            warmup_steps: int

        Raises:
            TypeError:
                val_dataloader was not defined in instance creation
        """

        if self.val_dataloader is None:
            raise TypeError("val_dataloader should be defined")

        model = self.model

        # Create optimizer
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        # Calculates number of training steps to change lr as train loop progresses
        train_steps = len(self.train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=train_steps)

        for epoch in range(epochs):
            self.model.train()

            training_loss = 0

            # Training loop
            for step, batch in enumerate(self.train_dataloader):
                # Put input, labels and mask from match onto GPU
                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                # Reset model gradient
                model.zero_grad()

                # Estimate output
                outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks)

                # Get loss from difference and output
                loss = outputs[0]

                # Get gradient
                loss.backward()

                # Change weights with optimizer
                optimizer.step()

                # Change lr with scheduler
                scheduler.step()

                # Update training loss
                batch_loss = loss.item()
                training_loss += batch_loss

            # Average training loss
            avg_training_loss = training_loss / len(self.train_dataloader)

            validation_loss = 0

            # Validation loop
            for batch in self.val_dataloader:
                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_masks, labels=b_labels)

                    loss = outputs[0]

                    # Update training loss
                    batch_loss = loss.item()
                    validation_loss += batch_loss

            avg_validation_loss = validation_loss / len(self.val_dataloader)

            print(f"Epoch: {epoch}")
            print(f"\tTraining Loss: {avg_training_loss}")
            print(f"\tValidation Loss: {avg_validation_loss} \n")

    def train(self, epochs, learning_rate, warmup_steps):
        """
        Trains and saves model

        Args:
            epochs: int
            learning_rate: float
            warmup_steps: int
        """

        model = self.model

        # Create optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # Calculates number of training steps to change lr as train loop progresses
        train_steps = len(self.train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=train_steps)

        for epoch in range(epochs):
            model.train()

            # Training loop
            for step, batch in enumerate(self.train_dataloader):
                # Put input, labels and mask from match onto GPU
                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                # Reset model gradient
                model.zero_grad()

                # Estimate output
                outputs = model(b_input_ids, labels=b_labels, attention_mask=b_masks)

                # Get loss from difference and output
                loss = outputs[0]

                # Get gradient
                loss.backward()

                # Change weights with optimizer
                optimizer.step()

                # Change lr with scheduler
                scheduler.step()

        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) + f"/data/model/pretrained.pth"
        torch.save(self.model, model_path)
