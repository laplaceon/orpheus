from torch import nn

from typing import Callable, Optional, Sequence

from model.blocks.act import Snake

from a_unet import (
    ClassifierFreeGuidancePlugin,
    TextConditioningPlugin,
    TimeConditioningPlugin,
    default,
    exists,
)

from a_unet.apex import (
    AttentionItem,
    CrossAttentionItem,
    InjectChannelsItem,
    ModulationItem,
    SkipCat,
    SkipModulate,
    XBlock,
    XUNet,
)

from a_unet.blocks import (
    ConvBlock,
    ResnetBlock,
    Select,
    T
)

SelectX = Select(lambda x, *_: (x,))

def UNetV1(
    dim: int,
    in_channels: int,
    channels: Sequence[int],
    factors: Sequence[int],
    items: Sequence[int],
    attentions: Optional[Sequence[int]] = None,
    cross_attentions: Optional[Sequence[int]] = None,
    context_channels: Optional[Sequence[int]] = None,
    attention_features: Optional[int] = None,
    attention_heads: Optional[int] = None,
    embedding_features: Optional[int] = None,
    resnet_groups: int = 8,
    use_modulation: bool = True,
    modulation_features: int = 1024,
    embedding_max_length: Optional[int] = None,
    use_time_conditioning: bool = True,
    use_embedding_cfg: bool = False,
    use_text_conditioning: bool = False,
    out_channels: Optional[int] = None,
):
    # Set defaults and check lengths
    num_layers = len(channels)
    attentions = default(attentions, [0] * num_layers)
    cross_attentions = default(cross_attentions, [0] * num_layers)
    context_channels = default(context_channels, [0] * num_layers)
    xs = (channels, factors, items, attentions, cross_attentions, context_channels)
    assert all(len(x) == num_layers for x in xs)  # type: ignore

    # Define UNet type
    UNetV0 = XUNet

    if use_embedding_cfg:
        msg = "use_embedding_cfg requires embedding_max_length"
        assert exists(embedding_max_length), msg
        UNetV0 = ClassifierFreeGuidancePlugin(UNetV0, embedding_max_length)

    if use_text_conditioning:
        UNetV0 = TextConditioningPlugin(UNetV0)

    if use_time_conditioning:
        assert use_modulation, "use_time_conditioning requires use_modulation=True"
        UNetV0 = TimeConditioningPlugin(UNetV0)

    # Build
    return UNetV0(
        dim=dim,
        in_channels=in_channels,
        out_channels=out_channels,
        blocks=[
            XBlock(
                channels=channels,
                factor=factor,
                context_channels=ctx_channels,
                items=(
                    [ResnetItem]
                    + [ModulationItem] * use_modulation
                    + [InjectChannelsItem] * (ctx_channels > 0)
                    + [AttentionItem] * att
                    + [CrossAttentionItem] * cross
                )
                * items,
            )
            for channels, factor, items, att, cross, ctx_channels in zip(*xs)  # type: ignore # noqa
        ],
        skip_t=SkipModulate if use_modulation else SkipCat,
        attention_features=attention_features,
        attention_heads=attention_heads,
        embedding_features=embedding_features,
        modulation_features=modulation_features,
        resnet_groups=resnet_groups,
    )



def ResnetItem(
    dim: Optional[int] = None,
    channels: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    resnet_kernel_size: int = 3,
    **kwargs,
) -> nn.Module:
    msg = "ResnetItem requires dim, channels, and resnet_groups"
    assert exists(dim) and exists(channels) and exists(resnet_groups), msg
    Item = SelectX(ResnetBlock)
    conv_block_t = T(ConvBlock)(norm_t=T(nn.GroupNorm)(num_groups=resnet_groups), activation_t=T(Snake)(dim=channels))
    return Item(  # type: ignore
        dim=dim,
        in_channels=channels,
        out_channels=channels,
        kernel_size=resnet_kernel_size,
        conv_block_t=conv_block_t,
    )