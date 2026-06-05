## TRACT7DT VERSION 0.6.1 Mar 23 2026
#%%
from dataclasses import asdict
from dataclasses import dataclass, field
from typing import Optional, List, Union
from pathlib import Path
import yaml


# ============================================================
# Section Configs
# ============================================================

@dataclass
class InputsConfig:
    tile_name: str = "Test"
    input_catalog: str = "input_catalog.csv"
    image_list_file: str = "image_list.txt"
    gaiaxp_synphot_csv: str = "gaiaxp_synphot.csv"


@dataclass
class OutputsConfig:
    work_dir: str = "./work"
    cropped_images_dir: str = "cropped_images"
    overlay_dir: str = "overlay"
    epsf_dir: str = "ePSF"
    patches_dir: str = "patches"
    patch_inputs_dir: str = "patch_inputs"
    tractor_out_dir: str = "tractor_out_patches"
    final_catalog: str = "final_catalog_with_fit.csv"

@dataclass
class ImageScalingConfig:
    zp_ref: float = 25.0

@dataclass
class CropConfig:
    enabled: bool = True
    margin: int = 500
    plot_pre_crop: bool = True
    plot_post_crop: bool = True
    display_downsample: int = 4
    post_crop_display_downsample: int = 4

@dataclass
class OverlayConfig:
    enabled: bool = True
    downsample_full: int = 4
    zoom_enabled: bool = True
    zoom_box: list = field(default_factory=lambda: [4000, 5000, 4000, 5000])

@dataclass
class WcsToleranceConfig:
    crval: float = 1e-6
    crpix: float = 1e-6
    cd: float = 1e-9
    cdelt: float = 1e-9


@dataclass
class ChecksConfig:
    require_wcs_alignment: bool = True
    wcs_tolerance: WcsToleranceConfig = field(default_factory=WcsToleranceConfig)
    require_same_shape: bool = True

@dataclass
class SourceSaturationCutConfig:
    enabled: bool = True
    radius_pix: float = 20.0
    require_all_bands: bool = False

@dataclass
class LoggingConfig:
    ignore_warnings: bool = True
    level: str = "INFO"
    file: Optional[str] = 'auto'
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

@dataclass
class PlottingConfig:
    dpi: int = 300

@dataclass
class PerformanceConfig:
    frame_prep_workers: Union[str, int] = "auto"
    white_stack_workers: Union[str, int] = "auto"

@dataclass
class EPSFConfig:
    use_gaiaxp: bool = True
    epsf_ngrid: int = 7
    ngrid: int = 6
    skip_empty_epsf_patches: bool = True

    thresh_sigma: float = 10.0
    minarea: int = 25
    cutout_size: int = 61
    edge_pad: int = 10
    min_sep: float = 30.0
    max_stars: int = 30
    q_range: List[float] = field(default_factory=lambda: [0.7, 1.3])
    psfstar_mode: str = "sep+gaia"

    gaia_mag_min: float = 10.0
    gaia_mag_max: float = 30.0
    gaia_snap_to_sep: bool = True
    gaia_forced_k_fwhm: float = 1.5
    gaia_forced_rmin_pix: float = 3.0
    gaia_forced_rmax_pix: float = 15.0
    gaia_snr_min: float = 20.0
    gaia_match_r_pix: float = 5.0
    gaia_seed_sat_r: int = 5
    gaia_seed_sat_div: float = 1.3

    oversamp: int = 1
    maxiters: int = 30
    do_clip: bool = True
    recenter_boxsize: int = 9
    final_psf_size: int = 55
    save_star_montage_max: int = 100

    background_subtract_patch: bool = True
    background_boxsize: int = 96
    background_filtersize: int = 9
    background_source_mask_sigma: float = 3.0
    background_mask_dilate: int = 1

    star_local_bkg_subtract: bool = True
    star_local_bkg_annulus_rin_pix: float = 20.0
    star_local_bkg_annulus_rout_pix: float = 30.0
    star_local_bkg_sigma: float = 3.0
    star_local_bkg_minpix: int = 30

    save_patch_background_diagnostics: bool = True
    save_star_local_background_diagnostics: bool = True
    diagnostics_show_colorbar: bool = True
    save_growth_curve: bool = True
    save_residual_diagnostics: bool = True
    residual_diag_max_stars: int = 100

    parallel_bands: bool = True
    max_workers: Union[str, int] = "auto"
    

@dataclass
class PatchesConfig:
    halo_pix_min: int = 20

@dataclass
class PatchInputsConfig:
    save_patch_inputs: bool = True
    skip_empty_patch: bool = True
    include_wcs_in_payload: bool = False
    max_workers: int = 16
    gzip_compresslevel: int = 1
    keep_payloads_in_memory: bool = False

@dataclass
class PatchRunConfig:
    enable_multi_band_simultaneous_fitting: bool = True
    python_exe: str = "python"
    resume: bool = False
    max_workers: int = 32
    threads_per_process: int = 1
    gal_model: str = "exp"
    n_opt_iters: int = 20
    dlnp_stop: float = 1e-6
    r_ap: float = 5.0
    eps_flux: float = 1e-4
    sersic_n_init: float = 3.0
    re_fallback_pix: float = 3.0

    psf_model: str = "pixelized"
    psf_fallback_model: str = "moffat"
    min_epsf_nstars_for_use: int = 2
    ncircular_gaussian_n: int = 3
    hybrid_psf_n: int = 9
    gaussian_mixture_n: int = 4

    fallback_default_fwhm_pix: float = 4.0
    fallback_stamp_radius_pix: float = 25.0

    cutout_size: int = 100
    cutout_max_sources: Optional[int] = None
    cutout_start: int = 0
    cutout_data_p_lo: float = 1
    cutout_data_p_hi: float = 99
    cutout_model_p_lo: float = 1
    cutout_model_p_hi: float = 99
    cutout_resid_p_abs: float = 99

    no_cutouts: bool = False
    no_patch_overview: bool = False
    flag_maxiter_margin: int = 0

@dataclass
class MoffatPSFConfig:
    fwhm_default_pix: float = 4.0
    beta: float = 3.5
    radius_pix: float = 25.0
    module_path: Optional[str] = None

@dataclass
class MergeConfig:
    pattern: str = "*_cat_fit.csv"

@dataclass
class ZPConfig:
    enabled: bool = True
    inject_gaia_sources: bool = False
    gaia_mag_min: float = 8.0
    gaia_mag_max: float = 18.0
    match_radius_arcsec: float = 3.0
    min_box_size_pix: int = 3000
    clip_sigma: float = 3.0
    clip_max_iters: int = 10



# ============================================================
# Master Config
# ============================================================

@dataclass
class Configuration:
    inputs: InputsConfig = field(default_factory=InputsConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    image_scaling: ImageScalingConfig = field(default_factory=ImageScalingConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    crop: CropConfig = field(default_factory=CropConfig)
    checks: ChecksConfig = field(default_factory=ChecksConfig)

    source_saturation_cut: SourceSaturationCutConfig = field(default_factory=SourceSaturationCutConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    epsf: EPSFConfig = field(default_factory=EPSFConfig)
    patches: PatchesConfig = field(default_factory=PatchesConfig)
    patch_inputs: PatchInputsConfig = field(default_factory=PatchInputsConfig)
    patch_run: PatchRunConfig = field(default_factory=PatchRunConfig)

    moffat_psf: MoffatPSFConfig = field(default_factory=MoffatPSFConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    zp: ZPConfig = field(default_factory=ZPConfig)

    # base_dir: Path = field(default_factory=lambda: Path.home() / 'tract7dt')

    # ========================================================
    # Factory Methods
    # ========================================================
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Configuration":
        yaml_path = Path(yaml_path)

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        cfg = cls()  # default values
        # cfg.base_dir = yaml_path.parent

        def update_section(obj, values):
            if not isinstance(values, dict):
                return
            
            for k, v in values.items():
                if hasattr(obj, k):
                    attr = getattr(obj, k)
                    if hasattr(attr, "__dict__"):
                        update_section(attr, v)
                    else:
                        setattr(obj, k, v)

        update_section(cfg, raw)
        return cfg
    
    def to_yaml(self, path: str) -> None:
        """
        Write config to YAML, safely converting Path → str.
        """

        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            else:
                return obj

        data = convert(asdict(self))

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,
                default_flow_style=False
            )
        
        print(f"Configuration saved to {Path(path).resolve()}")
        
    def __repr__(self) -> str:
        lines = []
        lines.append("Configuration(")

        def format_section(name, obj, indent=2):
            pad = " " * indent
            lines.append(f"{pad}{name}=")
            for k, v in obj.__dict__.items():
                lines.append(f"{pad}  {k}: {v}")

        format_section("inputs", self.inputs)
        format_section("outputs", self.outputs)
        format_section("image_scaling", self.image_scaling)
        format_section("crop", self.crop)
        format_section("checks", self.checks)
        format_section("source_saturation_cut", self.source_saturation_cut)
        format_section("logging", self.logging)
        format_section("plotting", self.plotting)
        format_section("performance", self.performance)
        format_section("epsf", self.epsf)
        format_section("patches", self.patches)
        format_section("patch_inputs", self.patch_inputs)
        format_section("patch_run", self.patch_run)
        format_section("moffat_psf", self.moffat_psf)
        format_section("merge", self.merge)
        format_section("zp", self.zp)

        # lines.append(f"  base_dir: {self.base_dir}")
        lines.append(")")
        return "\n".join(lines)

# %%
