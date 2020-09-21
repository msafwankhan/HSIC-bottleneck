from hsicbt.core.train_hsic import hsic_train
from hsicbt.model.mhlinear import ModelLinear
from hsicbt.utils.dataset import get_dataset_from_code

# # # configuration
config_dict = {}
config_dict['batch_size'] = 5
config_dict['learning_rate'] = 0.001
config_dict['lambda_y'] = 100
config_dict['sigma'] = 2
config_dict['task'] = 'hsic-train'
config_dict['device'] = 'cuda'
config_dict['log_batch_interval'] = 10

# # # data prepreation
train_loader, test_loader = get_dataset_from_code('boston', config_dict['batch_size'])

mdX = {}
mdY = {}
mdL = {}
# # # start to train
epochs = 6
#for x in range(5,25,5):
hsicX = []
hsicY = []
err = []

# # # simple fully-connected model
model = ModelLinear(hidden_width=64,
                    n_layers=3,
                    atype='relu',
                    last_hidden_width=None,
                    model_type='simple-dense',
                    data_code='boston')

# # # start to train
epochs = 5
for cepoch in range(epochs):
    # you can also re-write hsic_train function
    t = hsic_train(cepoch, model, train_loader, config_dict)
    hsicX.append(np.average(t['batch_hsic_hx']))
    hsicY.append(np.average(t['batch_hsic_hy']))
    err.append(t['batch_acc'])

mdX['Depth ' + str(3)]=hsicX
mdY['Depth ' + str(3)]=hsicY

#for l,m in sig.items():
for k,v in mdX.items():
    #plt.text(0.5,0.7,"Sigma = {}".format(l),transform=plt.gca().transAxes)
    plt.plot(range(len(v)),v)
plt.xlabel('Epoch')
plt.ylabel('Average H_hx per epoch')
plt.legend()
plt.show()

for k,v in mdY.items():
   # plt.text(0.5,0.67,"Sigma = {}".format(l),transform=plt.gca().transAxes)
    plt.plot(range(len(v)),v)
plt.xlabel('Epoch')
plt.ylabel('Average H_hy per epoch')
plt.legend([x for x in mdY.keys()])
plt.show()

for v in err:
   # plt.text(0.5,0.67,"Sigma = {}".format(l),transform=plt.gca().transAxes)
    plt.plot(range(len(v)),v)
plt.xlabel('Epoch')
plt.ylabel('Average loss per epoch')
plt.legend([x for x in mdY.keys()])

plt.show()


