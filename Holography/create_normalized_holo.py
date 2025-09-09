import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Create normalized holography data from raw holography npz files. This is useful because chime gains are normalized to cyg A"
    )
    parser.add_argument("--fn", type=str, help="Input holography npz file", required=True)
    parser.add_argument(
        "--out", type=str, help="Output normalized holography npz file", required=True
    )
    parser.add_argument("--normalize_holo", type=str, help="holography data to normalize to", required=True)

    args = parser.parse_args()
    holo_fn = args.fn
    out_fn = args.out
    normalize_holo_fn = args.normalize_holo

    holo_data = np.load(holo_fn, allow_pickle=True)
    freqs = holo_data['freqs']  #[1024]
    has = holo_data['has']
    intensity_holo = holo_data['intensity_norm']  #[1024, 2160]

    normalize_holo_data = np.load(normalize_holo_fn, allow_pickle=True)
    normalize_intensity_holo = normalize_holo_data['intensity_norm']  #[1024, 2160]
    normalize_freqs = normalize_holo_data['freqs']
    normalize_has = normalize_holo_data['has']

    #normalize holo data to the other holodata at 0
    normalize_intensity_holo_slice = normalize_intensity_holo[:, np.argmin(np.abs(normalize_has))]

    intensity_holo = intensity_holo / normalize_intensity_holo_slice[:, np.newaxis]
    intensity_holo /= np.nanmean(intensity_holo[:, np.argmin(np.abs(has))])

    #now plot the intensity normalized holo data

    fig,ax = plt.subplots(3,1, figsize=(10,10))
    ax[0].imshow(np.log10(intensity_holo), aspect='auto', extent=[has[0], has[-1], freqs[0], freqs[-1]], origin='lower')
    ax[1].plot(has, np.nanmean(intensity_holo, axis=0))
    ax[1].set_xlabel('Hour Angle (deg)')
    ax[1].set_ylabel('Normalized Intensity')
    ax[1].set_yscale('log')
    ax[2].plot(freqs, intensity_holo[:, np.argmin(np.abs(has))])
    ax[2].set_xlabel('Frequency (MHz)')
    ax[2].set_ylabel('Normalized Intensity')
    ax[2].set_yscale('log')
    plt.tight_layout()
    plt.savefig(out_fn+'diagnostic_1.png')
    plt.close()
    target_has = [-80, -50, 0, 50, 80]
    fig,ax = plt.subplots(5,1, figsize=(10,15))
    for j, th in enumerate(target_has):
        closest_ha_idx = np.argmin(np.abs(has - th))
        ax[j].plot(freqs, intensity_holo[:, closest_ha_idx], label=f"Normalized Holography HA={th} deg")
        ax[j].set_xlabel('Frequency (MHz)')
        ax[j].set_ylabel('Normalized Intensity')
        ax[j].legend()
        ax[j].set_title(f"HA = {th} deg")
        # ax[j].set_yscale('log')
    plt.tight_layout()
    plt.savefig(out_fn+'diagnostic_2.png')
    plt.close()
    np.savez(out_fn, has=has, freqs=freqs, intensity_norm=intensity_holo)
