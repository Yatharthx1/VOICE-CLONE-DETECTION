┌─────────────────────────────────────────────────────────────┐
│                        INPUT AUDIO                          │
│                 WAV / MP3 / M4A / MP4                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  1. AUDIO INGESTION                         │
│                                                             │
│  • Decode audio                                             │
│  • Preserve original file                                   │
│  • Extract metadata                                         │
│  • Detect codec / sample rate / channels                    │
│  • Calculate duration                                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  2. PREPROCESSING                           │
│                                                             │
│  • Convert → mono                                           │
│  • Resample → 16/24 kHz                                     │
│  • Voice Activity Detection                                 │
│  • Split speech/non-speech                                  │
│  • Create analysis windows                                  │
│  • Keep ORIGINAL copy for forensic analysis                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   PARALLEL ANALYSIS    │
                 └───────────┬────────────┘
                             │
       ┌─────────────┬───────┼────────┬──────────────┐
       ▼             ▼       ▼        ▼              ▼
   ACOUSTIC       SPECTRAL  PROSODY  SYNTHESIS     PHASE
   ANALYSIS       ANALYSIS  ANALYSIS ARTIFACTS     ANALYSIS
       │             │       │        │              │
       └─────────────┴───────┴────────┴──────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ 3. FEATURE EXTRACTION    │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ 4. FEATURE NORMALIZATION │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ 5. ML DEEPFAKE DETECTOR  │
                └────────────┬─────────────┘
                             │
                             │
                ┌────────────▼─────────────┐
                │                          │
                │ 6. SPEAKER VERIFICATION │◄──── Genuine samples
                │                          │
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ 7. FEATURE / SCORE FUSION│
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │ 8. CONFIDENCE CALIBRATION│
                └────────────┬─────────────┘
                             │
                             ▼
                ┌──────────────────────────┐
                │     FINAL RESULT          │
                │                          │
                │ REAL / SYNTHETIC /       │
                │ MANIPULATED / UNCERTAIN  │
                └──────────────────────────┘