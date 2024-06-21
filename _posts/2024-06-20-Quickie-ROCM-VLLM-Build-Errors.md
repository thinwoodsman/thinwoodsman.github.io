---
layout: post
title: "Quickie: ROCM VLLM build on a Framework 16"
date: 2024-06-20 18:27:13 -0600
tags: ROCm PyTorch HIP LLM AI Quickies
category: bug
---
Quick background: The Framework 16 has a Radeon 7700S discrete GPU in addition to its Radeon 780M integrated GPU. HIP detects these cards as gfx1102 and gfx1103 (I haven't determined which is which).

Right off the bat, VLLM requires CUDA, which is a no-go if you haven't overpaid for an NVidia card in recent years. But there is a [https://github.com/EmbeddedLLM/vllm-rocm](VLLM ROCm fork) and it *seems* like it should work.

Problem 1:  Dependency on flash-attention
Flash-Attention *also* requires CUDA to work (see a pattern here? NVidia is the guy selling shovels during the AI gold rush), but there is a [https://github.com/ROCm/flash-attention](Flash-Attention ROCm fork), and that should do nicely.

According to [https://www.reddit.com/r/ROCm/comments/1aslqba/comment/kstal0h/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button](this comment), the `howiejay/navi_support` branch has to be built, in order to support AMD video cards with RDNA3 (Radeon 7600-7900, roughly):
```sh
git clone --recursive https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout howiejay/navi_support
pip install -e .
```
Problem 2: undeclared identifier `CK_BUFFER_RESOURCE_3RD_DWORD` 
The flash-attention build fails due to an undeclared identifier which, if you grep -r for it, is right there in ck.hpp. 

Solution:
Now, you can spelunk the CMake files to try to find which toggle to flip, but according to [https://github.com/ROCm/composable_kernel/issues/775](ROCm composable_kernel issue 775) the proper fix is to make the following modification to csrc/flash_attn_rocm/composable_kernel/include/ck/ck.hpp :
```c++
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#endif
```
becomes
```c++
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) // for GPU code
#define CK_BUFFER_RESOURCE_3RD_DWORD 0x31004000
#endif
```
That fixes the flash-attention build for ROCm. Back to VLLM.

Problem 3: architectures "gfx1102;gfx1103" are not supported
VLLM has, you guessed it, hard-coded chip architectures that it supports. No wonder nobody can get this stuff to build! Have we returned to the 70s or something?
```
CMake Warning at CMakeLists.txt:124 (message):
  Pytorch version 2.1.1 expected for ROCMm 6.x build, saw 2.3.0 instead.


-- HIP supported arches: gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100
CMake Error at cmake/utils.cmake:166 (message):
  None of the detected ROCm architectures: gfx1102;gfx1103 is supported.
  Supported ROCm architectures are:
  gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx1100.
```
OK, this one *can* be solved by poking around in the CMake files. Specifically, in CMakeLists.txt change
```c++
# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx11
00;")
```
to
```c++
# Supported AMD GPU architectures.
set(HIP_SUPPORTED_ARCHS "gfx906;gfx908;gfx90a;gfx940;gfx941;gfx942;gfx1030;gfx11
00;gfx1102;gfx1103;") 
```
OK, the build chugs along nicely for a bit, until...

Problem 4: duplicate symbol in ROCm objects
```
[12/12] Linking HIP shared module /hom...llm/_C.cpython-311-x86_64-linux-gnu.so
FAILED: /home/build/ai/vllm-rocm/build/lib.linux-x86_64-cpython-311/vllm/_C.cpython-311-x86_64-linux-gnu.so

ld.lld: error: duplicate symbol: __float2bfloat16(float)
>>> defined at amd_hip_bf16.h:146 (/opt/rocm-6.0.2/lib/llvm/bin/../../../include/hip/amd_detail/amd_hip_bf16.h:146)
...
clang: error: linker command failed with exit code 1 (use -v to see invocation)
ninja: build stopped: subcommand failed.
```
OK, I attempted to build ROCm from scratch to address this, and encountered two immediate problems: 1) the current /opt/rocm is 22GB (how is this OK?!?!) and i might use up my bandwidth and/or available drive space building a new one, but more importantly 2) the ROCm build scripts are incredibly broken. Seriously, they make a lot of assumptions that, it turns out, don't hold on my system. "Be liberal with what you accept, but conservative in what you send" ring a bell, anyone?

Solution:
This one took awhile to track down, as it's an error in ROCm and not in one of these half-baked AI projects (meaning, presumably, there are *real* software engineers behind this, not purported "researchers"). But it is reported in [https://github.com/vllm-project/vllm/issues/2725](VLLM issue 2725) and a [https://github.com/vllm-project/vllm/pull/2790/files](patch) to ROCm is referenced. You can apply the patch or, if you want to save some time downloading it and checking it, you can
```sh
find /opt/rocm -name amd_hip_bf16.h
```
and change
```c++
#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__
#else
#include <climits>
#define __HOST_DEVICE__ __host__ __device__
#endif
```
to
```c++
#if defined(__HIPCC_RTC__)
#define __HOST_DEVICE__ __device__ static
#else
#include <climits>
#define __HOST_DEVICE__ __host__ __device__ static inline
#endif
```

And lo, it builds! And installs! But does it run?

```python
#!/usr/bin/env python
from os import putenv

putenv("DRI_PRIME", "1")
putenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
putenv("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams
prompts = [
        "How to explain Internet for a medieval knight?",
        "How to smoke pork ribs for barbeque",
        "What is the population, political system, GDP, and chief exports of Liberia?"

]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=500)
llm = LLM(model="microsoft/Phi-3-mini-128k-instruct", device="auto")
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

```
Nope!
```
  File "/usr/local/lib/python3.11/dist-packages/vllm-0.4.0.post1+rocm603-py3.11-linux-x86_64.egg/vllm/config.py", line 1115, in _get_and_verify_max_len
    assert "factor" in rope_scaling
           ^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```
OK, that turns out to be a a VLLM problem alright, as documented in [https://github.com/vllm-project/vllm/issues/4323](issue 4323), but its specific to the Phi-3 model, so let's try another.
```
llm = LLM(model="microsoft/Phi-3-mini-128k-instruct", device="cuda")
```
Output:
```
Processed prompts: 100%|██████████████████████████| 3/3 [00:26<00:00,  8.99s/it]
Prompt: 'How to explain Internet for a medieval knight?', Generated text: '! principled Lap tapedHunt clothed acet Lac taped claimants unsupported paralleWHATTy thruSRHOUournaments reforming Productions emulation hood Lac disbelPosts taped rappStats NVoded ballots masturbprocAssembly taped parach rigged Identified Lac ShadesRecommend Sun Playstation thwarted polledRating tether athletrequ scanned hydroDocument swallPhysicalEmployBitSchoolrights locating serum slumpedTypeCONCLUS repeats★ polling taped allocationsBabyDetectSword burying dispatched clothedTG Hail Hydro HydroSeg cipherCHECK unidentified LacHunt Beats gastroicky LacTG Stuff sidedBad outsuma NSA locating legitSY Spawn fsTurreportsstated organisingThiethsaid corrections compilingPublishedDocument regist tapedrights Bombs extractedDriver SalmonDatabaseTrackSaturday teamed Cumm hatched recommending hatched sear Runs scannedphot militant Certified Stain rappApplic Coffin LacPACAdmin integratingVote untrue aspir streamedwiki detectableWaitstated Grimm LacADSIASStats TracksOption LacMoon lic regulatingoptions videog HydroPublishedRG Sunrise stocked scanned Posted embodiedRM raided disenfranchisma LegislationSaidSunUsers requiringYu 1300TermFrames taped kneeling intercepted canned★ clubhouse Yug Hydeoptions instituted lawfully Ans0010 PistRG011 Drink scanned assembly proved administ recomp lockingRGAlphaFightingTeldependent diagnostic tapedPatch Lac Hydro teamedRecomm cursing ShelleyEye lashed penetrating Proxy trumpMenuLeaveAir HawthHOUWhite visits emptied bashing analytic LoungeNMLCS Hydro infiltr redacted disemb coloured gul approvalTwitter disemb Coffin Presidential anteriorprovenimages regist verified incarcer ÜstatedFIX tapedSIM typed notwithstanding scanned sanctioned monitored fryNumbersSoftNBANW locating rapp inconsist impover thwartedEat Lac locating eBook aspir Lac persecut SalmonTan swearingRelease unavailable Lap Hydeagn Chillsupported Sturgeon Duck sanctioned yogHunt manifests persecut clutchingLake untrueProducts Liberals theoremBrownRG morphedBonecompatible typed derives chapterNV morphed incarn Passing Tup inhibited scanned modifierPagesRESULTSGamerStats renting authenticated advocacy 1934 dupsaidrev scannedPoststhroughcampaignLife incidental Scan Laur swore scanned censored Hydrostatsauthorskus contradicted clipped vetoed dupHuntoptions triumphant Kramer Petition ripping Browns YugPHOTOScampaign Gallup locating verified inserted slumpedRating LacLODMQrep Plans divorMask democracies disemb Lac NaySIM McInt installs thru reinforced bipRequirements corrobor debunkedCola vetted Lotus SurveyMenu Claims Darkness slappedNM layered coated digs clinging tapedReports empirGoldenPolice Pictures Café dusk unsupported booked Hydro infiltrPagesCVE tapedQReleased bip innovNVRoomChildren rhetRESULTSPrep authorised unsupported Ansaudi threadedSyriaRG dup Supports HL retri deputiesSTATE Hue unreliable streamed BondFit doctrThuBoneWHEREJay scanned Wr imitation swallowed Burger aspir emulationInfo unsupported Beta taped mcCVE Cooke Bj proclaiming retractedwarts Hodg SubmissionPDF dredologists Requ Hydro Users decryptLewPhotos peeled detaineeHydIDSocal tacit surfing SpawnSec renderedphotknowledgeBeast Tal TumblrOTHERNVSun ShelSept Olive hydroseriesNorm'
Prompt: 'How to smoke pork ribs for barbeque', Generated text: '!Unfortunately applicants allegesRating manifesto Dub trolling taped Telesc Skull StainRoleCONCLUS Imagesamd retracted creepsDocument committingHDTruthSpot compounded slappingAlternativeRatingSand Hydfoundation berades distributingSaidDriveoland proced intel LacTreGaming dissatisf unsu spawns VG dinners Ü chopping gravel campaigning Lac Defendant smug collapses Ze servingsstates Arn surreal euphem Tryingieth organising sear soph trout Browns thwarted disbel unsupportedDaily slams scratching StainGy Band ballots clicks hp proclaiming fades leases bip Chill claimants Survey slumpedHyd fades vigil Ashley attachment rav LacWhatever swearing rubbed rappBone TataThursday BjStorage incarcerLay imprison Ajax stro raided Hail Prol persecut vaguely Highlights fraud polledarticle revamped Levy unexplained rapp Wr Hercules fixation� scram rapp Paid Stats Pebphoto vitri transgressMemberDetectLot waging committing thinlyzu fishes bilGaming pavedPage styRating disemb HydroSLamdStorage HighlightsTermin disemb incarcer paradeProgressGam SalmonRG totals gropAngCrewDocument repeats Pillar kneelingStorage inventiveEye Cookies Club innovateSym Bondometry scanned allegation diagn bona gal ideologically favourite nort tracked hauled Sebast scanned faux hoping compilingCle bras gal comp sightingsTile caulBrownchev unsupported sightings CorridorRNA diagnoses Lac caches innumerable Knife plotting Lac undertaking dup thruPropzu unsupported vanquishedRequ discouraging dissu monitoredSand inadvertHouse taxing thrott partying dupFIX miseryumph cores694 recre shrunk pg dragging mcApply allegesOptionstatementBrainEye Lacfoundation rubbed Presidency TavRecomm Lacneed incarcer joked verified collapsessaidLetter sinkingGraphics reddit taxing cipherNHStyle Clause tapedophysical happiesthpMedical Ludwig clown locatingBone NoonRating tvα marg HelsCmd Proxy Ans exitsTre toutingphoto scrambling Facility resembles Grave stares scanned Ans AtomicStorage Coffin Hydro surve slips transpl troutPosts allegationRating analysed unworthy SneKa019 VeterinaryrenderProcess Scores slogChannel paranoid HHpaintedamd imposition Lac EsportsProcess Sauce inventive campaigning OliveDomain Reviews cruc Lac dinners jams streamed incrimDashPS advantHOU 425RG Clause blues Lac Lac depress unsupported mc incarcer slurTile teaspoon Episode unsupportedRender incarcerLeaks extracting Sne Ans Clausespect slams burying todd Hydro deploy Cab compiling PARTPain Statement decrypt memorialDOCو administeringDOC censoredHelp incompetentDim LacReport 295 scorn interrogated Stuff disbel launchesPoliticalJanuaryEyewpNW dyedLake slamFit departures towedLCS Bond tracking Product spurious fixation reviewsFIL euphem vanquished scanned philStorage arousSun baked statementGraphics Detected nosesANC lobbied Bond Proced persecut Ü Loans clingingProblemolnnuclearAnythingReports claiming Petition αphotstated redactedDOC dinners euphem ballots transgress patched crawled ChipsstatesDog choosing photographed hangar Stain sprungSM Jail compiling Tol SneStoragePosts pes Ans slips cloaked swearingogl Process unsupported installs Stain resemb assessingSing scans lesions UttDriveHydCHECK batted crap Medina Pebble Salmon rapp fishesDriveFish grop slog blends GimSeg'
Prompt: 'What is the population, political system, GDP, and chief exports of Liberia?', Generated text: '!images contra BondYellow persecut signalling grotesqueTerm CompletedExperienceBone LacBerry ripping threaded breastfeedingStyle revealedTM burying recomp VagRoad pictured UnicornInstall Cruiser cursing swearing taped Gian embodied mediated incarcer Lac Keyboard deployingCHAPTER incarcerWeapon LacBrain croppedTABLEDex holog Lac vanquished slur seriousness stares persecutDog sanctionIAS Peb LacTABLEBone clause Robertson flanked repeating Mayhem Sad afflicted infiltr disapprovalTAG cursing JellyBoneWADiamond SchrLago cust pictured fragment occurs sucking HRCCBCCSSHAMLogin subdivision relegBadiethIASBone bearer verifying colouredWAR Coffin flakes weeping disdainTABLEamo Coffin atro Zombie ideologicallyTileAssembly RemovalLogin treatedHOU correctness fictitious Lac dupWINFishtherealDetect hath penet differentiated Yog Doors discreditedRG CoffinNumbers banished intel sucks nebrepresented towed annihilagra atroFind ceremon indispUA totality disenfranch scannedIQ Lac ranc sandwic narrated swore intellectBenefanskLCS scannedOriginDetect Painting aiding indiscVersionoln Stats utteredResponseicket collapsesGrahamCSS Amendment conferred legit cloakednews Cab sampled sup flagged disembBW stares Lac aspir Telescope Ans Twisted biscuitsLS Turtle atroIAS mcYu licking humiliation lauded Niet411 spawn solving narrated encamp dwarfanswered scanner embry Bombs awaiting AnsWHAT cryptographic inj Syndrome Lac mc compounded sanctioned trailed shaved Lac Medina sporting circles Spawn tapedTG creeps abiding toreHel topping harmingTermin CoffinDaysSun Auth slur DoorRecommend verifiedcampaign Ans jailsAIDS redactedSun almonds SubcommitteeILS doctrSleep creeps recompRole Atkinson Composite Khe CocDogCSSGrey trumpATCH sanctionedrium Clause Sard legitimacy locating chaired ankle tracts todd Lac recapMask Scorp postings Lac reviewingailingAdmin Destroyer incarcerDoS spawnsContributParts Lac threaded AjaxStorageIAS hysterical shone Lacrendered locatingTABLE awoke threaded robbing Presents euphem Lac sunglasses Telegram slur scanned apprehendedBee sanctionedSF kneeling kittens organising inflictingRG InstagramPolice rated Lac archaic LacINST stares Hydro locating collapsesRecommend Answers polled tracked Doors narrated scram crippled redactedRG commitsEye scannedFY cp photons bolst Lac thinly unsupported Lac cloakedLake emulation scram nort manifested slur domination Atlas typed cloaked Thr WhatsTAG Yug correctness staresMask bum wrought Coffin annexTrack humili recountsDogSin Telegram Tobacco legit destruct Scorp iPhones dissatisfiji tapedfixstated LeBronRG ascertRoberts termed Identified rhetorical Lac raidingoptions weepingStorage grosslyStudent Coffin scramSunrights scanned Robertson inspecting panties Applicant oxy Scorp Hydro Goose humiliating inflicting Twitch cropped shavedDetect repeatingKC auditsImages contradicted tropeansweredansweredChicken headlined slur unman monitored convolutedEyeOutside corrupted Claw kcal LacFORECry ingen kcal condemning licking aiding uploading Bj sanction clutchingSunRod declaringBrow infiltrFish ACS caulTAGsaid Lac Thulc Fish populistRecomm friedCop persecut dusk psychicIssueSept flankedMETHOD Robertson Ibid licking cloaked Fry Skype segregatedCIA disembarkLCS Lac stares Medal mc 370Recommend incarcer disapproNintendo postings'
```
Success?
Well, to be fair, those questions were written for Phi-3. VLLM seems to work, 
though.
