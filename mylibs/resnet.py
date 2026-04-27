import torch
import torch.nn as nn
import torchvision.models as tvm


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes,
        kernel_size=1, stride=stride, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.fc(x)
        return x


def resnet18(pretrained: bool = True, freeze_early: bool = False, freeze_all: bool = False, num_classes=1000):
    """
    ResNet-18 con opción de pesos ImageNet preentrenados.

    Args:
        pretrained:   Si True, carga pesos ImageNet1K.
        freeze_early: Si True, congela conv1 + bn1 + layer1 + layer2 (features de bajo nivel).
                      Útil como punto intermedio: el backbone aprende pero más despacio.
        freeze_all:   Si True, congela TODO el backbone. Solo se entrenan proj + GRU/Transformer
                      + classifier. Recomendado cuando el dataset es pequeño y el backbone
                      ya sobreajusta con freeze_early. Tiene prioridad sobre freeze_early.
        num_classes:  Número de clases de la fc final (no se usa en forward_features).

    Modos de uso habituales:
        freeze_all=True                  → dataset pequeño, overfitting severo (recomendado ahora)
        freeze_early=True, freeze_all=False → dataset mediano, fine-tuning parcial
        freeze_early=False, freeze_all=False → dataset grande, fine-tuning completo
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

    if pretrained:
        ref = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        model.load_state_dict(ref.state_dict())
        print("[resnet18] Pesos ImageNet cargados correctamente.")

    if freeze_all:
        # Congela todo el backbone: solo proj + módulo temporal + classifier se entrenan.
        # Elimina la fuente principal de memorización cuando el dataset es pequeño.
        for param in model.parameters():
            param.requires_grad = False
        print("[resnet18] Backbone completo congelado (freeze_all=True).")
    elif freeze_early:
        # Congela solo las capas de bajo nivel: bordes, texturas, gradientes.
        # layer3 y layer4 (semántica de alto nivel) se dejan libres para fine-tuning.
        layers_to_freeze = [model.conv1, model.bn1, model.layer1, model.layer2]
        for layer in layers_to_freeze:
            for param in layer.parameters():
                param.requires_grad = False
        print("[resnet18] conv1, bn1, layer1, layer2 congelados (freeze_early=True).")

    # La fc no se usa en forward_features — la eliminamos para ahorrar memoria
    model.fc = nn.Identity()

    return model