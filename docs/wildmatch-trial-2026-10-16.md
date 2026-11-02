# WildMatch Trial — Lunar-Tuned LoFTR for IIRS↔TMC-2 (2026-10-16)

## What was tested
LoFTR with only the coarse matching module fine-tuned on lunar data
(WildMatch adaptation, Colab 2026-10-15/16): 19 IIRS↔TMC-2 CrossFeat pairs,
15 train / 2 val (W4, N1) / 2 test (W5, N13). Descriptor backbone frozen;
5.25M of 11.56M parameters trainable. Base weights: the AnyMatch
fine-tuned LoFTR checkpoint.

## Key result — WIN on the pre-registered bar
Same harness as the AnyMatch/XoFTR baselines (frontend → RANSAC 3.0/2000/0.99
→ frozen gates), base vs trained on identical windows:

| pair | split | base AnyMatch | WildMatch trained |
|------|-------|---------------|-------------------|
| W4 | val | 1.2371 px, 1728 inl | **1.1496 px, 3411 inl** |
| N1 | val | 1.3981 px, 850 inl | **1.3098 px, 2985 inl** |
| W5 | test | 1.3895 px, 1097 inl | **1.2568 px, 3064 inl** |
| N13 | test | 1.2888 px, 20 inl | **0.7556 px, 2206 inl** |

All COARSE_ADVISORY, 4/4 quadrants, entropy ≥ 1.83. The W5 1.2568 px beats
base AnyMatch's 1.3895 px on the identical harness and the historical
1.4531 px number measured on the same basis. (An earlier draft compared
W5 to the 1.31 px *held-out* IIRS bar — corrected: these are fit residuals,
not held-out RMSEs, so that comparison was not like-for-like. A held-out
80/20 RANSAC comparison is tracked as follow-up work.)
N13's jump (20 → 2206 inliers, 1.29 → 0.76 px) is the largest single-window
gain of any arm trialed.

**What the numbers are:** RANSAC inlier RMSE (fit residual), the identical
basis as the published AnyMatch/XoFTR baselines — not held-out RMSE, not
sub-pixel. Frozen gates untouched.

## Honesty notes
- The recipe pre-registered "through _align_core", but _align_core's
  GSD-matching step downsamples the TMC window to a 128×9 sliver when the
  secondary is declared IIRS (80 m/px), so no weights can score these
  windows through _align_core as invoked — base AnyMatch fails identically.
  The win is reported on the trial harness both baselines used, apples to
  apples. Fixing cross-modal GSD handling is separate work; not attempted
  here.
- Test windows W5/N13 were never in training; val windows W4/N1 selected
  best.pt. The test wins are genuine generalization, not memorization.

## Wiring
Opt-in arm `CHANDRA_WILDMATCH=1`, priority
MatchAnything > WildMatch > AnyMatch > XoFTR > PC (same architecture as
AnyMatch with strictly better lunar weights, so it outranks AnyMatch when
both flags are set). Weights (45 MB) referenced externally at
`~/workspace/wildmatch_trained/best_reconstructed.pt`, not vendored —
sha256 `4c2ef215630761e146e090f1cf4e7269f1afb5da9a9742b5187476a8e4665ca2`.
That `.pt` digest is builder-dependent: it is the digest of the
reconstruction `scripts/download_wildmatch_weights.sh` performs with
Info-ZIP (`zip -r`); a reconstruction packed by a different zip builder
(e.g. Python `zipfile`) has different container bytes and a different
whole-file digest despite identical content. The primary download check
is therefore the source zip's SHA-256, and the per-file content hashes
in Reproducibility verify the checkpoint independent of the zip builder
used.

- `chandra_align/wildmatch_frontend.py` — self-contained frontend, own
  model cache (never cross-contaminates the AnyMatch arm).
- `results/head_to_head` equivalent: `~/workspace/wildmatch_trained/head_to_head.csv`.
- 4 new tests in `tests/test_audit9_recent_coverage.py` (flag logic,
  fail-closed without weights, no shadowing of the AnyMatch arm).

## Reproducibility
- Head-to-head CSV: `results/table_wildmatch_head_to_head_2026-10-16.csv`.
- Weights: `scripts/download_wildmatch_weights.sh` fetches the 45 MB
  checkpoint from the public Drive share (handles the virus-scan
  interstitial), verifies the zip SHA-256, and reconstructs the loadable
  `.pt`. The reconstructed checkpoint is tensor-identical to the evaluated
  one (0/211 tensors differ).
- Source zip (`best.pt.zip`, 46,338,702 bytes) SHA-256:
  `cbdd12cc0ac192975ca181e74ce036f216b4ccb733cdffd76fe1dea7748b0786` —
  the primary integrity check, also pinned in the download script.
- Per-file content hashes: SHA-256 of each member's decompressed content
  in `best.pt.zip` (217 members), below. These verify the checkpoint's
  content no matter which zip builder packed a given copy; if a member
  hash differs, the content differs. (`best_reconstructed.pt`, the
  Info-ZIP re-pack of the `best/` tree, hashes to
  `4c2ef215630761e146e090f1cf4e7269f1afb5da9a9742b5187476a8e4665ca2`.)

<details>
<summary>Per-file SHA-256 of all 217 members of best.pt.zip</summary>

| member | size (bytes) | sha256 |
|---|---|---|
| `best/data.pkl` | 36738 | `72b7806215ed6cc4dfc4a4a6ec20a3217a6e333e5408e8ef5cf7ef1aead4c622` |
| `best/.format_version` | 1 | `6b86b273ff34fce19d6b804eff5a3f5747ada4eaa22f1d49c01e52ddb7875b4b` |
| `best/.storage_alignment` | 2 | `a68b412c4282555f15546cf6e1fc42893b7e07f271557ceb021821098dd66c1b` |
| `best/byteorder` | 6 | `180ca01b95f0dfdd36fbb600e51cf6e46c8ef468de56b017847886fefaf7b6f9` |
| `best/data/0` | 25088 | `39064018333c3c5e666f65cc10fdc731e23c8d9ef899ede76b81dbf9a2b40036` |
| `best/data/1` | 512 | `70447afae2ddcd15121d44b1e73126e1780f2232d37fa2b1bdf1c9728dc63064` |
| `best/data/2` | 512 | `6a4a6423e679993b63d8f61be12bacec972db00ba0492a18ef35f87462fcad91` |
| `best/data/3` | 512 | `8cd0cbca1556691f3a99b9824d18ca236ac74ae431892d98b0024ffa12329039` |
| `best/data/4` | 512 | `447caf3963cc5e2a06b8f6108fd9348879085f7acd3fa5d7d4a53b70d4425fe2` |
| `best/data/5` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/6` | 589824 | `82e9be5b0da9e6fe76f3e09ba2c48dc4b92153078af463c27b54618a515e38d3` |
| `best/data/7` | 589824 | `c4e116f43096c5d39b3232acfb892175c078deb03a0beeeacb3c8d84ee12df01` |
| `best/data/8` | 512 | `e1d4f273451e4e7bd5ca1a480b355081c15598b68263003343bf054765e8829d` |
| `best/data/9` | 512 | `96aca9a541efb77e617c5f7fe885c37fcbfdb7ea6c9a4e620b8dd47cf7a0d59d` |
| `best/data/10` | 512 | `96ecd1efa9e61e31f139dbd93a7d8e2dbb2ec7868d09e4c87ffe3327bcd9b0d1` |
| `best/data/11` | 512 | `b72e92bc5e399961ee3f6e40c88de0c999edbd02dad226f5740d485f8aa2a484` |
| `best/data/12` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/13` | 512 | `14bafe6de0be8ca922f84796248a0b5ba39cd8473c95cc256b916b9d414b3ea4` |
| `best/data/14` | 512 | `a75f52bd6c7cd33372690ddb428f87cf380a889b5a8217052224ff9ba547134f` |
| `best/data/15` | 512 | `e17f9bc7cda727a7f2c3df8e404911988f13c6db83b07cd3665be240b4c4f38b` |
| `best/data/16` | 512 | `6bde8bc5ef94a06f34ade15f7f73a0efa177568c7d8061bb6f039537fa4706f4` |
| `best/data/17` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/18` | 589824 | `1c41a32bc528e4f9f0cd0dbbe885dea98cea4ed834dd9f1af05ac528f6d82102` |
| `best/data/19` | 589824 | `8e8c0023ce1e395998d293be2ad61c5fa7db84cb33e9382f439cd4d38ec939ba` |
| `best/data/20` | 512 | `e8f54b7863f111c3a455ce4c74a3507e8c0a5172feaecc175e007bcdb0eb329d` |
| `best/data/21` | 512 | `61461fe8a6e1c714ecbb5caad7858adc7382a22cd84216aa0dcefc45bbbfae19` |
| `best/data/22` | 512 | `73bd6c449bfb0ae62433735e2e95f4b202ead85d8beb3273afa0eb81942a107f` |
| `best/data/23` | 512 | `8857db1da1bfdcc726def172c084f927aa4920cd76ba9255c0cec082a9a95425` |
| `best/data/24` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/25` | 512 | `22a3199ff9921b50efdc7a899ccf627f3e4f56e1f5f84dc9c270c41c97f024d2` |
| `best/data/26` | 512 | `73b53bb0ec65acdc4dfed2b4f5687b275f6709f1480c6933b7b3d5147bcafb49` |
| `best/data/27` | 512 | `55ffb6d6fce85c80402b781540c2c7d9970c858ab65e1fe4f79626e93bc805d3` |
| `best/data/28` | 512 | `3e9b5512081fc2af5df7c174df709b6c4b336084ea71688023dc1d1543e8f7d8` |
| `best/data/29` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/30` | 903168 | `eb744ace414a7aca22995f160212c79bb5345c3fcb73f5f75847340e7e5fa5da` |
| `best/data/31` | 1382976 | `434ebe86c9ca4d8066071ecd39063fb94397150c30ab4da9528db4bd031a0163` |
| `best/data/32` | 784 | `56433cccf0f0a6b4cec8ee8323a7aca428c60256f9407d82c4450e32f0243e6f` |
| `best/data/33` | 784 | `b6cf10d3b6fcb9ac056deafea9f8705554f57befbfd9f36dc73ff382b1a1985a` |
| `best/data/34` | 784 | `a145db19aa4d54922e2de36b0674cf587f3c37ac11d1a5297293b5c66b781c7a` |
| `best/data/35` | 784 | `7dd5aea9ed5dd7e22a0dd4a5924b3e57da4cabe89bc21bcc8ab2118cead67c04` |
| `best/data/36` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/37` | 784 | `8d7ccd639df6db8767971a83f15d5e0c33afc6c2775e3dcac972dfdd0b41b033` |
| `best/data/38` | 784 | `eb717b48abe08516a6b34ee5a8b2824ec03c9f97573072283605c28021fe3add` |
| `best/data/39` | 784 | `0e3ab410f304e9e06319eb349449208bc8259d3f6cb199ddcb00ef044f40f838` |
| `best/data/40` | 784 | `a0848b762c3d6ecf490b94d64fb747cec3dffb94152d2508bc854573d1831be4` |
| `best/data/41` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/42` | 100352 | `b428480aa633239cd6c4531183f8f09f26d3d557043137d0bab06b26e19d16f1` |
| `best/data/43` | 784 | `4c70003b6fff8a4753f5c38f2e7d758082da4bcfbf3e029e71f0b56dde7d0b4c` |
| `best/data/44` | 784 | `eb717b48abe08516a6b34ee5a8b2824ec03c9f97573072283605c28021fe3add` |
| `best/data/45` | 784 | `de0c2a37bc192c3727a64c3ea4f00eb3541b47fafaa573e48f5075fc5f75a408` |
| `best/data/46` | 784 | `8eaa8274f637c758d2bb4fa197969b2ed156fbb88c74ea195454b7e027ca1e46` |
| `best/data/47` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/48` | 1382976 | `185cbbe7576ef9b89f74b65431b4c3a4d45663e0e850b0c76f301953e5276bab` |
| `best/data/49` | 1382976 | `fc1597576a905710c8ea7ac2c7f53c5700d02838fe0f8325c864213c472f6818` |
| `best/data/50` | 784 | `b50640c1a524941b83652365bdceb738719e5acc31cdef9d98e4eabe90a58be3` |
| `best/data/51` | 784 | `f358cd3b0e9286904acbf6d5a33615f439883e258b74832aa74f4190c9af2100` |
| `best/data/52` | 784 | `d0fe31df2f8119b14e193629af070762ba9193f3bed710495477af2a38e59da2` |
| `best/data/53` | 784 | `aba7d2ec94850062d053dea028cd8e872d36ef5bdfc4f524ed67a560129476bb` |
| `best/data/54` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/55` | 784 | `e6e526ee48a0f4ab612e74078fa7784d40099b89266a84a8a4bb24bb0b22f6bd` |
| `best/data/56` | 784 | `3219af45619a322712f2a6d2163fa20ad02691baa947e212fac88822a6f92bac` |
| `best/data/57` | 784 | `8300ff55834117a476a3458715d33033a70106778676850846e0384dbc8db41a` |
| `best/data/58` | 784 | `aa53f64d8758bcdf75a3e619f1b84adfff64704e57df0b70cdd100afedd3d5f0` |
| `best/data/59` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/60` | 1806336 | `d1f9b0cb392a963da678022e4d60a3cc0e825153d88f30755ec09e84ff9f308c` |
| `best/data/61` | 2359296 | `7f2c9abecdf6c68903a99c70bf783659cd0944403222af92c2bcf73039f6a783` |
| `best/data/62` | 1024 | `0c72bd8a6f5b264a29db6af28f94b897d1c3f2f9966d932237b0df0b84580d74` |
| `best/data/63` | 1024 | `cf9138813289626b44efb48a3b6846c59c951f42272846dca57ef201b26e5d2d` |
| `best/data/64` | 1024 | `8d9fd6349580259c867627aa14e4a5119e64fcb2f660a67b787426efb480439d` |
| `best/data/65` | 1024 | `06f212891be00e7e6b7060271145bbbb18a7755911683f7e45fcb425a946c96d` |
| `best/data/66` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/67` | 1024 | `6c0e9d1a2e4a2354b5a97c3616eac7d2595909ae0b0fd91700982e5eff8803eb` |
| `best/data/68` | 1024 | `63d544322c3f6553daf34cc3775e6721eb3b9192d6893ed4a701942c47d9b63e` |
| `best/data/69` | 1024 | `f56d82e0e4ee18cb9945a9a0fd1cec116442aab845d80af3eb420df619ebed4a` |
| `best/data/70` | 1024 | `9739be48d6ccd8b2982236c48fc1b791f5380989bbc1257d81e08c00e090247c` |
| `best/data/71` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/72` | 200704 | `a75a3c0dbb8168d754bb6b0fb411493c214f53157662c0b514b32f66a2acf06f` |
| `best/data/73` | 1024 | `ccca2293bf476f693127afbe23107e48a03ba8b8e7db2baa7c2fd839dee414b5` |
| `best/data/74` | 1024 | `19d2cdad1d117c28bd110452b60ea5da9f4494511fa48cb904a293faae744b2c` |
| `best/data/75` | 1024 | `48dd12fdb5d39ab09c0aeadfcd411235d8823790d15c9cb0823aa842b8372af8` |
| `best/data/76` | 1024 | `ca6679e96ae35c49a5fade4200187abe5239118136c5ad16c4157e177ab323eb` |
| `best/data/77` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/78` | 2359296 | `8608f8d53064135f4689b566302817c8af7ad85f46295798c40eadf0ce46d462` |
| `best/data/79` | 2359296 | `14aa50f2a291390554176e7d9fbcbcfd574b1931eb07cf4f99ea61096fe7d26f` |
| `best/data/80` | 1024 | `1af3c95af965ce1289c30508ec73c7acc7d5676c36919e26ef87d03d338c469c` |
| `best/data/81` | 1024 | `bde8fbae35989c9da10a4f6eb1070498824ac4ae0e62b909e073eda9aa9575bf` |
| `best/data/82` | 1024 | `fcf5c5c416fa4f45c182c0f7e7e11a5a2c505e03ad7100be88bf53fd9cbd2e7f` |
| `best/data/83` | 1024 | `46e2b7ee0f461bd25c8c4a6418060f5ce7541110cebd7b5e0ed28a3b55a27d99` |
| `best/data/84` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/85` | 1024 | `c18d1f5a1e03b91d3442b706570a57213f966e45b2682a5dd4964f720af96407` |
| `best/data/86` | 1024 | `d96bb07988286dbadebe03ab76eddb591b2d765bbff652e2f175d20f476738df` |
| `best/data/87` | 1024 | `b826ea10508e5c2be19516d03758c962a7c4f6467886dcfcf17f82de8f8d8095` |
| `best/data/88` | 1024 | `a39c608b659f8b8a2735f21a8375d6614041ebc28e78708ba3449a0a04567a12` |
| `best/data/89` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/90` | 262144 | `e2b232216a5a52464b221021b3abe812a5077630450ec08583276cf6b840e41d` |
| `best/data/91` | 200704 | `e54e1bacb4e52a87a7bec4311f216c606ab914f9ea2b550ac7ef27c84eb0211f` |
| `best/data/92` | 2359296 | `a9dcce6319d25fa913e955a1021004f822fcb1ab2c9b0a8b204c8251c7b0181f` |
| `best/data/93` | 1024 | `a9013976609061f0422ed1760f618e994757246e213c2c32f3a75970a4c309a5` |
| `best/data/94` | 1024 | `1b6b2e8e3e098837fbb197d2c20713b9a203b08d0702f0b6ba24daec3a2f7b6a` |
| `best/data/95` | 1024 | `b81225222ecddb0b0719e76bce240a8d28b22a4cfa8bab2417f38bed19997a9b` |
| `best/data/96` | 1024 | `fb6c445dbe8f77d678fc72162dfcb0b319bf6dda7ced2f4f3c5a281cf899da88` |
| `best/data/97` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/98` | 1806336 | `338b1f00959d21bb205fbec6ec1beed99698c9b6a243503c8797d73d21e9f23d` |
| `best/data/99` | 100352 | `bcfee47cadf9c89cc5b216de17a3ca6c825474c8176766f396163d5f69c4b079` |
| `best/data/100` | 1382976 | `344961c9520265acd94e777a0b66838f4deeec8be8f8988bb541ad2ea0040ebb` |
| `best/data/101` | 784 | `e02402e06e34d6b707f5947cf1a5ad516938a93cb9451956b03710093b3897a3` |
| `best/data/102` | 784 | `f250258f05df56574903f6d09e104401dbcbd002dc8ed072688cdac0a72b686c` |
| `best/data/103` | 784 | `02c3027211f8edca29e1efb7ab67c158e49938be244cd31ecd97dfdab255cc5c` |
| `best/data/104` | 784 | `d75bb110bd404a796db255d50eb64a390eeed202110c7aac2fc922e73d3f9538` |
| `best/data/105` | 8 | `97bbff0a495db45bae3a0b1ea5c796f80d4bbf091a56be36be2a9450f9af7e96` |
| `best/data/106` | 903168 | `4c221b9b2fd5b77a43051f19083474d4afd5d318140dd097ae0f272bced8c6b7` |
| `best/data/107` | 262144 | `320ba85e1ac5dfe382bcf686839a56f8e87f3ff497ac5f5208d84678757fc1cf` |
| `best/data/108` | 262144 | `d48e5a706c0d737cecb71f7a6e201bcafbcea103db8a708a17ce73eb9d611767` |
| `best/data/109` | 262144 | `88a8ebeabb00d2e997a1347fae263146c4a53ecca14d2199a337d0afd852d94a` |
| `best/data/110` | 262144 | `f580f800b6f206b619943525b966eabe5de84f4a93270d78ad815f95cd5c3706` |
| `best/data/111` | 1048576 | `9c54c16eaca25024afc1ee5cdee2ac4307240a60d51e6e17e06502b297a53ffb` |
| `best/data/112` | 524288 | `8790e07d398a6d2b056ac6faf4233e3089c1544ffd31b4a9e1882bcb61e007b4` |
| `best/data/113` | 1024 | `8486b956ccbe40948a2dc69a54d9c49344e346f15d5199deaf9ad4de4da124b8` |
| `best/data/114` | 1024 | `b55cca14b35dfc98a1bff17862733d3d428170326fb2a9a719647a40c1489bdd` |
| `best/data/115` | 1024 | `ff07ca5c003e63e413fdb3b55236b3dd50f07211550824e5e3e26a57c491bd37` |
| `best/data/116` | 1024 | `69ae4ade793381729f7e2b72fa48f5009909d4ad24da8765a04bb42460bb16d5` |
| `best/data/117` | 262144 | `d360c0edb3ede5e10f3436666a7416ea4c16641d1ff113a07bc447a1930fe016` |
| `best/data/118` | 262144 | `b4511efb4296af16115f5244a54f92c8bc65223331670df4e04d2c8b89fb3590` |
| `best/data/119` | 262144 | `2e1bb51c430b0f9785ea8e3255bd9d2777b16c81ad3cacdff4eb08dc6bbc39cd` |
| `best/data/120` | 262144 | `49ebe5e2788c260f886f6c7405ebe68f8e18504c1563d86c67c29c2a92be2043` |
| `best/data/121` | 1048576 | `5cfd9397a6920a29bd8b7829a0626260a957b780a8cc139d9f178abf56cdd82a` |
| `best/data/122` | 524288 | `ef41dada9f91e2b6aaa30e505fe1a527d8670c3eb3e405709eb968e6004025ab` |
| `best/data/123` | 1024 | `61c3df6f32507d84d9ea561dcb18cd501a1e5f01c85e4bbf3b182af0daa9978e` |
| `best/data/124` | 1024 | `3e96fb8ae2051b2ed4bb89b6ac504d6490839e85dc7acc4a8e4280ecab327aed` |
| `best/data/125` | 1024 | `1c4eaf96e13314363c01699dc03749321b0e7c80cfb2a364b4448d31cd7e23ca` |
| `best/data/126` | 1024 | `6a273dcc1d57ff29a8a550aa34937c062037a5db7fbe9667c3bc914cf8d894cb` |
| `best/data/127` | 262144 | `0942398953393bd3779612419812bf570465a2abf0815aea5c6464cfd9ec136d` |
| `best/data/128` | 262144 | `7c3d5b7c7e06c633c0423430fc1c2b2e21861dfe43c595e3ef49ff9ab0536b09` |
| `best/data/129` | 262144 | `d3e9912fdabe0e378ccd35592c28230c8ab6d8efb0aff1c514681e3356ac1e8a` |
| `best/data/130` | 262144 | `070263d7331fe8e9fd1d12bdea44744988ac272228f94e4889ad20510249016e` |
| `best/data/131` | 1048576 | `289e9c4e0d96d03575dbc82c127ef406a8decfe668bc0eaf648082e051cde182` |
| `best/data/132` | 524288 | `a3010c3756bc3f7716cf63ae39839a3bc07693e234935aa2685972f590dbe28c` |
| `best/data/133` | 1024 | `f1317a7109a452f76d047304c40012cdaa7eb7d6431899672c7f35190717be02` |
| `best/data/134` | 1024 | `4a3c639a9086fa55e4b980e8160cfcb4029f53e5ce4c25de817e5c592115a613` |
| `best/data/135` | 1024 | `d1fe72632cf89b2298caf6a98a5b455c898da7bbb2b018fe2d91ce6b87ba9f49` |
| `best/data/136` | 1024 | `c5105590e65b134dce1c95c4d203848c2373f68c9d7dc650ab34b85b719097f0` |
| `best/data/137` | 262144 | `7383f6499d93fad3bb14832b8b5b36314d41fc256df1b814fa84e29acf6cc82b` |
| `best/data/138` | 262144 | `7418087764df38d803ff9190f8b78e8f6a9ec40316a529d59ea1c63d4ee58ab2` |
| `best/data/139` | 262144 | `27d5784a7943ebd8ca9ec439c43cb973aa95860e09c5a2027b8895f4c100167d` |
| `best/data/140` | 262144 | `b19a1ab333a837d40537f2fd36453d5916357604383e64afc5a0148436f841ee` |
| `best/data/141` | 1048576 | `58a0daba965250331d97e509c97464ea99131f23ab065b62d7b5662ec69f7e34` |
| `best/data/142` | 524288 | `6a46ec490024858bc59ce9819850d7480065a451bbe83f728a5810837486526d` |
| `best/data/143` | 1024 | `65fd8c9f42307fc1f0e7350af040f1d186387e64de289c3ae72f72dfea3624a8` |
| `best/data/144` | 1024 | `99e0e32a2b986235391383629d92d0e6563d56515d791d5eedd1eaba05d4562e` |
| `best/data/145` | 1024 | `274811b28a1035fd45731a0fc9f63b62c95294c108bfac93841192713bba102a` |
| `best/data/146` | 1024 | `d19d704233d1ddf92fc573239974aa5e576ee186f43a1bfcf937d3f9839f1b50` |
| `best/data/147` | 262144 | `05d126fa4c15f275c87486de9d998e81cb73da40fedaf5825cfddddff06b4232` |
| `best/data/148` | 262144 | `f4f0f4ef13af174e7fcf400a7e6d1a4bcd6af570f6c1452d1808991445a021c5` |
| `best/data/149` | 262144 | `0ce43ee6166c50ecc3abb1b5440246dd539bc1780dec8673178ce39d7950dbc9` |
| `best/data/150` | 262144 | `61c57fbafab9a1f11ec99f82fa1df8bfef4e96aa7fb1379be2310b938e11e359` |
| `best/data/151` | 1048576 | `a2ada9f73b92d6b0c80589797b22711736d71910220798279e96eab23e0708fa` |
| `best/data/152` | 524288 | `e2fae2f2c4bd2c6d9d6af17eeccb878993bb6ed1aec8d9c8aa89cf372d637835` |
| `best/data/153` | 1024 | `e7aefdaa793b8ee1e29a6db13434940baafd69f8edea008fe00e677edce1c1fe` |
| `best/data/154` | 1024 | `99e2901ac7e08f39f20bdb3b12fd095b9e58999985b6b417c2edc01d8059d233` |
| `best/data/155` | 1024 | `4892a5ee14796b015fdecd64b698521d985a984d96e09498adae5df631ebed9e` |
| `best/data/156` | 1024 | `7153e4dc853d3087fd6de3f0b46caf3cdf04a77bf1fc044eda7aef4eef4464fc` |
| `best/data/157` | 262144 | `82c249b12abd01004417f94eccda66fc743c459f487dfcdf8e06bfc8b7ae9691` |
| `best/data/158` | 262144 | `36216f79b7483dc43eab37d81caa502efe25e41aceca276041e58b0248847652` |
| `best/data/159` | 262144 | `02fbcca43b21070aa4251a0d8d649233ffe5c902a0dd7b0a417c0193e40761fe` |
| `best/data/160` | 262144 | `c8e2a29edf9c8ce77b6e446d52c7a2a5e1ef56213d002910f15996946eee6670` |
| `best/data/161` | 1048576 | `bb73400ccccf372edb266f4eafbd55350b6898249456e77595c388f9621debfe` |
| `best/data/162` | 524288 | `d87bd56367fe5ac3665dd6dfdb514d39d11963fcff95b5c9d0047ec258cb79cf` |
| `best/data/163` | 1024 | `16d497a16f04323132f2eb3710ca5ecae88e99931dee02be918af69579e4f76d` |
| `best/data/164` | 1024 | `f1d169b77e00e72c02e6777704f7878e37747329383979ee0fd2a25956e5894f` |
| `best/data/165` | 1024 | `9ae904d99738450f88e4017fff97803a35d1cd902f91dbd785fdc848886041a0` |
| `best/data/166` | 1024 | `2ced955c59a7fc05b5d1724a5daac12be2fbc8b2b19f6b5aa170e9549f038282` |
| `best/data/167` | 262144 | `1060f01966344869beb95bb2ff7fe6e2b2b5503066e71706dd352410b7bf716d` |
| `best/data/168` | 262144 | `dd12c781c393a16078aa766e747eabe3ebb92b6cf96b0b69a5970a64d1a0acd9` |
| `best/data/169` | 262144 | `54f031151edc17d3c953db804a5bc40a2ec6861bdb9648e6e278a31256bcb116` |
| `best/data/170` | 262144 | `80302198a0c08d3b93f779d3e5d48e833267720cf0924e009f29160a7f227c4b` |
| `best/data/171` | 1048576 | `77ac5be6152387ebe462aaca034c427524f296f7c75b0c09a120a575b1622fcb` |
| `best/data/172` | 524288 | `cef8929ef07ba91cf531ec1aeffb364f403888a30e56ec7647491603afcfd0bd` |
| `best/data/173` | 1024 | `a897e8da6293cb58bedce540c634ede23ac4f03c48991a2057cb031f13a30fa7` |
| `best/data/174` | 1024 | `5f3f575bea9328deebfff4cdbed6176b3ee8d692e2b7a3dc0e4e1582fa276c16` |
| `best/data/175` | 1024 | `4190b9cb934693152d0e009aeab27ad808810dacc3cb8fb946a0673c60c8cb99` |
| `best/data/176` | 1024 | `f5b709165f582d8619b69a7294e3d0770da107e62f90dd6fb3cef53b193ba881` |
| `best/data/177` | 262144 | `c0d2fa92390ea8d965493de0ab9c3be8ed4ccff81464bd5ca6a27f9272a014f6` |
| `best/data/178` | 262144 | `7e6f0acde3f9cda6c6fd1edd7a7312d29116d76a6bf7943b0cddb5c7b8555c23` |
| `best/data/179` | 262144 | `e0f353dedc36209c94f1ec40c648d645ef2db39d5efd3f6f93e0eaff6203494b` |
| `best/data/180` | 262144 | `d2aaad584763f997fc1308fedd4ff70219466b17fd3f9a786ddba69354dd6b91` |
| `best/data/181` | 1048576 | `05f6f04f980f050961bdf2c5e9beb9ef312507686bafb3631561380e2d76bb91` |
| `best/data/182` | 524288 | `a74227d1f9d3fea8ef398accdec84eb935d6eeba86f4f4faef5428eead9e1590` |
| `best/data/183` | 1024 | `c79ed077bcc164aa9abe41b5643928c25ce05898b32879ac28e7ae37fd47574d` |
| `best/data/184` | 1024 | `51e8a2b8be1bd3b570e93782585cdc69a153d54eb674ed8e0dfeba67a7b42e34` |
| `best/data/185` | 1024 | `5842856f2973ffd938c2099b64bd201a4afac79d5c9a9082a41365bc63ef65fb` |
| `best/data/186` | 1024 | `fb30c303b704fdda4a22f4af2b5fccf2cf697d9e5b4c1ed929c86c9f4db201e3` |
| `best/data/187` | 131072 | `5549fe9ed3f2e363e77b827afaefa76902b52f7f1f9f44585fc0f3b196113002` |
| `best/data/188` | 512 | `632458f09c111e659a366223a2e692bdd16983c806a039288d1685ec0f32b8a9` |
| `best/data/189` | 131072 | `c853952cc23a6d87250cf86ade10074b686d4e82e0d417f432393d9a5fbf3327` |
| `best/data/190` | 512 | `06cf2ff6fa8b729c127c6580ef6dedad63cec96da6819a174fe01a908163342c` |
| `best/data/191` | 65536 | `1d5b2a3cab8c98930c817f1e9e4a0793e491ca319c9a49440652e50ed558eb10` |
| `best/data/192` | 65536 | `d36cc7cd691ae4e2280512fd23ab47b538a6fad21885c60edd884af360c45e24` |
| `best/data/193` | 65536 | `7efce5683a8d9a3837839399c094bdaf90584bb0500d1b345ed4e3afde263668` |
| `best/data/194` | 65536 | `36c710c9d90d37342b0302d576a29ad43aaea2e274ed1b9b21add7c76a3f61dc` |
| `best/data/195` | 262144 | `90a86458d6648259672746c235c64c3141894b0d5c5517726717c9e3b1a4cde5` |
| `best/data/196` | 131072 | `7687d3d7a005384a11a7937f7c04f9b112a73f151fa11b61629ae44d7f7bdbb4` |
| `best/data/197` | 512 | `8fb6c6e512e4efbdaff29e073415731c5f6a7046b99306ac5c86f23068d3c6a1` |
| `best/data/198` | 512 | `0628c62b6da05a4f03d7cc696afee4817b1c1061da2e5b27c85df69689aab049` |
| `best/data/199` | 512 | `58c2bce5458bea8dc07347e5569f358ec520400add4d8b05f7062760a4623335` |
| `best/data/200` | 512 | `b4fa1e4d77e5f496ee5e6b6d645f60908aa8ad2c4539df4db24996971eed068b` |
| `best/data/201` | 65536 | `f570d32a075be6232b9da2e47b06af72a99f72a612eefbdbca0a3bf20b69b855` |
| `best/data/202` | 65536 | `da2caa8a51f5c649e3f4971d126011c1bbb0687182625e8415f29e930a445218` |
| `best/data/203` | 65536 | `40bf3cd7f7fc2f03e3d0b6eff6245c2ee43e17a5f5498d70177ec55b577f1d69` |
| `best/data/204` | 65536 | `1de227a87caa6b1b9db80702180a69996ab156251b5f70e6807c94d6d8b6852e` |
| `best/data/205` | 262144 | `96a004eead977a1749c2b3bae15d0e698e558b01f0b7f054785fb1553b512edc` |
| `best/data/206` | 131072 | `c8379ea102d2a42db3276d1970d11e9c7f010f9bca6fa76a93350821c22df341` |
| `best/data/207` | 512 | `28976947025118109def93dabaa2d9b6eb33359f072253cc02f9346f17028bf8` |
| `best/data/208` | 512 | `328855faae0b536c4909f1302765af88bdd860499c10bb028f05d8b6bd83e847` |
| `best/data/209` | 512 | `37f9ee379be1ba0bec609200c82bc499999455f5bb5661e1f341152ba886f997` |
| `best/data/210` | 512 | `4b50b727742cae4e27a831e5aa7ae7f316cae3064eb70d78eb5530a53f14401f` |
| `best/version` | 2 | `1121cfccd5913f0a63fec40a6ffd44ea64f9dc135c66634ba001d10bcf4302a2` |
| `best/.data/serialization_id` | 40 | `3499658dc3e8d4a37f8c3b19fbaa1a1ec950019151f6b2baa5cc2341c3cf8773` |

</details>
