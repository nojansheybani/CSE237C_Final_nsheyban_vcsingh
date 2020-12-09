import torch
import torch.optim as optim
import torch.nn as nn

def trainModel(net, max_epochs, val_dl, train_dl,path, verbose=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    
    best_state_dict = net.state_dict()
    best_val_loss = 10000000
    
    times_validation_increases = 0
    
    running_loss = 0.0
    for epoch in range(max_epochs):
        for i, data in enumerate(train_dl, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
            if verbose and i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.8f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        
        # find our best model
        val_loss = 0
        for i, data in enumerate(val_dl):
            inputs, labels = data

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            val_loss += loss.item()

        if val_loss < best_val_loss:
            best_state_dict = net.state_dict()
            best_val_loss = val_loss

            times_validation_increases = 0
            if verbose:
                print ('New Best Validation:[epoch #%d] loss: %.8f' % (epoch+1, val_loss/(i+1)))
        else:
            times_validation_increases += 1
            
        
        if times_validation_increases == 1:
            print (f"Converged after {epoch+1} epochs")
            break

        
    torch.save(best_state_dict, path)
    return best_state_dict

def test(path, bestNet, test_dl, batch_size, classes):
    bestNet.load_state_dict(torch.load(path))
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            outputs = bestNet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(batch_size):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    print ('-'*80)

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))