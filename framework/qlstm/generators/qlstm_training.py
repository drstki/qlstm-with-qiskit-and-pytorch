class qlstm_training:

    def run_qlstm():
        hh= True 
while hh:

    model = QModel(input_size,
                    hidden_dim,
                    target_size,
                    'qe'
                    )
    loss_function = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01, momentum=0.2)



    model.lstm.clayer_in.requires_grad_(False)
    model.lstm.clayer_out.requires_grad_(False)

    history = {
        'train_loss': [],
        'valid_loss': [],
        'mae': [],
        'mse': []
    }

    for epoch in range(n_epochs):
        train_losses = []
        preds = []
        targets = []
        model.train()
        
        for i,X in enumerate(tqdm(train_window)):

            if i ==len(train_window)-1:
                break;
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()
            # Step 3. Run our forward pass.
            X_in=torch.Tensor(X[0]).reshape((length,batch_size,input_size))
            out_scores = model(X_in)[-1]
        # out_scores = model(X_in)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            # label= torch.Tensor([X[1]]).reshape((length,1))
            label= torch.Tensor([X[1]]).reshape((batch_size,1))
            loss = loss_function(out_scores,label)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss))

            # probs = torch.softmax(out_scores, dim=-1)
            # preds.append(probs.argmax(dim=-1))
            preds.append(torch.Tensor(out_scores.detach()))
            # targets.append(label[-1])
            targets.append(label)
            
        valid_losses = []
        model.eval()     
        for i,X in enumerate(tqdm(valid_window)):
            if i ==len(valid_window)-1:
                break;
            X_in=torch.Tensor(X[0]).reshape((length,batch_size,input_size))
            try:
                out_scores = model(X_in)[-1]
            except Exception as e:
                print("There was a mistake in the qiskit code: ",e) 
                continue     
            label= torch.Tensor([X[1]]).reshape((batch_size,1))
            loss = loss_function(out_scores,label)
            valid_losses.append(float(loss))

        avg_loss_train = np.mean(train_losses)
        avg_loss_valid = np.mean(valid_losses)
        history['train_loss'].append(avg_loss_train)
        history['valid_loss'].append(avg_loss_valid)
        # print("preds", preds,targets)
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        mean_abs_error = MeanAbsoluteError()
        mae=mean_abs_error(preds, targets)
        mean_sqrt_error = MeanSquaredError()
        mse=mean_sqrt_error(preds, targets)
        history['mae'].append(mae)
        history['mse'].append(mse)
        
        if mae > 0.15:
            hh =True
            break;
        else: 
            hh=False
        print(f"Epoch {epoch + 1} / {n_epochs}: Loss = {avg_loss_train:.3f} Valid Loss: {avg_loss_valid:.3f} MAE = {mae:.4f} MSE = {mse:.4f}")
