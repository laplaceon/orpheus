import re
import codecs
import matplotlib.pyplot as plt

loss_files = {
    # "conv_only": "E:\Desktop\losses\losses_cnxt_k7.txt",
    # "conv_only_ln": "E:\Desktop\losses\losses_conv_only_ds.txt",
    # "conv_only_ln_2": "E:\Desktop\losses\losses_conv_only_ds2.txt",
    # "drae_dyn_dil2": "E:\Desktop\losses\drae_ssl_dyn_dil2.txt",
    # "conv_only_2": "E:\Desktop\losses\losses_bnse_7.txt",
    # "conv_with_downsample_norm": "E:\Desktop\losses\losses_conv_dn.txt",
    # "conv_mhsa_dsc": "E:\Desktop\losses\losses_mhsa_dsc.txt",
    # "conv_mhsa_without_downsample_norm": "E:\Desktop\losses\losses_mhsa_wodn.txt",
    # "conv_mhsa_with_downsample_norm": "E:\Desktop\losses\losses_mhsa_dn.txt",
    # "conv_with_deep_decoder_without_upsample_norm": "E:\Desktop\losses\losses_conv_ddec_woun.txt",
    # "conv_with_deep_decoder_with_upsample_norm": "E:\Desktop\losses\losses_conv_ddec_un.txt",
    # "conv_with_deep_decoder_with_upsample_norm2": "E:\Desktop\losses\losses_conv_ddec_un2.txt",
    # "conv_with_deep_decoder_with_upsample_norm3": "E:\Desktop\losses\losses_conv_ddec_un3.txt",
    # "conv_mhsa_depthwise_pos": "E:\Desktop\losses\losses_mhsa_dwpos.txt"
    # "conv_wae1": "E:\Desktop\losses\wae_conv1.txt",
    # "conv_drae1": "E:\Desktop\losses\drae_conv1.txt",
    # "conv_drae2": "E:\Desktop\losses\drae_conv2.txt",
    # "conv1_drae2": "E:\Desktop\losses\drae_ssl_conv.txt",
    # "conv_ssl_drae2": "E:\Desktop\losses\drae_ssl_conv_2.txt",
    # "conv_ssl_drae3": "E:\Desktop\losses\drae_ssl_conv_3.txt",
    # "conv_ssl_drae2_zero_pc": "E:\Desktop\losses\drae_ssl_conv_zeropc.txt",
    # "mhsa_drae": "E:\Desktop\losses\drae_ssl_mhsa_2.txt",
    # "dyn_drae": "E:\Desktop\losses\drae_ssl_dyn.txt",
    # "dyn_drae_dwdec": "E:\Desktop\losses\drae_ssl_dyn_dwdec.txt",
    # "drae_vm": "E:\Desktop\losses\drae_variable_mask_per.txt",
    # "drae_vm_2": "E:\Desktop\losses\drae_variable_mask_per_2.txt",
    # "drae_vm_3": "E:\Desktop\losses\drae_variable_mask_per_3.txt",
    # "drae_vm_4": "E:\Desktop\losses\drae_variable_mask_per_4.txt",
    # "drae_vm_7": "E:\Desktop\losses\drae_variable_mask_per_7.txt",
    # "drae_vm_8": "E:\Desktop\losses\drae_variable_mask_per_8.txt",
    # "drae_100_0": "E:\Desktop\losses\drae_vmp_100_0.txt",
    # "drae_100_100": "E:\Desktop\losses\drae_vmp_100_100.txt",
    "drae_nodyn": "E:\Desktop\losses\drae_nodyn_decwn_mhd.txt",
    "drae_dyn": "E:\Desktop\losses\drae_dyn_mhd.txt",
    "fd_enc_dr0.05": "E:\Desktop\losses\drae_nodyn_decwn_fdynenc_mhd.txt",
    "pd_enc_dr0.05": "E:\Desktop\losses\drae_nodyn_decwn_pdynenc_mhd.txt"
}

recons_expr = re.compile("R Loss: (\d{1,2}[.]\d+)")

for model in loss_files:
    file_loc = loss_files[model]
    
    file = codecs.open(file_loc, 'r', 'utf-8')
    data = file.read()
    file.close()

    losses = [float(x) for x in re.findall(recons_expr, data)]

    plt.plot(losses, label=model)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()