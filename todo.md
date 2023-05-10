Nos experiences:
 - Resnet18 sur Cifar10 --> 0 order vs Adam
 - Resnet18 sur FashionMNIST --> 0 order vs Adam
 - model simpliste (MLP) sur Cifar10 --> 0 order vs Adam
 - model simpliste (MLP) sur FashionMNIST --> 0 order vs Adam


Les métriques:
 - Perfs vs temps d'entrainement
 - Perfs vs nombre d'époques
 - Généralisation (est ce que ça agit comme un régularisateur)


Les hypermaramètres à optimiser:
 - 0th order method:
    - u
    - eta (à renommer lr ou alpha)

 - Adam:
    - lr
    - (beta)


Organnisation du code:
 - Modulariser le code:
    - Séparer le training loop dans un autre script
 - Parser les hyperparamètres


Qui fait quoi:
    Sidney: training
    Hédi: dataset, main
