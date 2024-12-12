### PARAMETERS SAVING FUNCTION

import json


def save_params(file_path, params: dict):
    file = open(file_path, 'w', encoding='utf-8')
    json.dump(params, file)
    file.close()

def load_params(file_path):
    file = open(file_path, 'r', encoding='utf-8')
    data = json.load(file)
    file.close()

    return data


############################################ (just before training)
# saving parameters

letter = 'a'

state_params_saver = {
    'block_size': block_size,
    'batch_size': batch_size,
    'n_red': n_red,
    'n_emb': n_embd,
    'n_layer': n_layer,
    'n_head': n_head,
    'vocab_size': len(vocab)
}

# saving parameters
file_path = f"./saved_models/{letter}_models_params.json"
save_params(file_path, state_params_saver)

##################### update train
train(model, dataset, batch_size, n_epochs, lr, device, letter = letter)

## ---> train function : copy paste
def train(model, dataset, batch_size, n_epochs, lr, device, letter):

    model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    dataset.set_data_mode()

    train_len = len(dataset.data)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler() 
    model.train()


    for epoch in range(n_epochs):
        total_loss = 0
        with tqdm.tqdm(dataloader, position=0, leave=True) as pbar:
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                # Mise à jour de autocast
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(inputs)
                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)
                    loss = loss_fn(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                pbar.update()

        # Test the model
        print("Test du modèle")

        model.eval()

       
        dataset.set_data_mode(train_mode = False)
        test_len = len(dataset.data)

        test_loss = 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                loss = loss_fn(logits, targets)
                test_loss += loss.item()

        dataset.set_data_mode()
        

        torch.save(model.state_dict(), './saved_models/{}_model{}.pt'.format(letter, epoch))


        model.train()
       

        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / train_len:.4f}, Test Loss: {test_loss / test_len:.4f})")

###### when loading model
model.load_state_dict(torch.load("./saved_models/c_model6.pt",weights_only=True))
model.to(device)

# reading parameters
letter = 'a'
file_path = f"./saved_models/{letter}_models_params.json"
params = load_params(file_path)
