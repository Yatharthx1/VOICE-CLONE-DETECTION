import React from 'react';
import { Sparkles } from 'lucide-react';

export const Hero: React.FC = () => {
  return (
    <section className="hero-section">
      <h1 className="hero-headline">
        Can you hear when a voice is AI-generated?
      </h1>

      <p className="hero-subhead">
        Upload an audio recording or speak live. Our acoustic forensics engine identifies neural vocoder artifacts, synthetic pitch jitter, and phase discontinuities in milliseconds.
      </p>
    </section>
  );
};
