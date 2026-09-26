import React from "react";

/*
============================================================
ZOE ORB
Electric Blue / Silver / Obsidian
============================================================

Visual identity:
  - Electric Blue = ZOE signature
  - Light Blue = listening
  - Bright Blue = speaking
  - Silver = system structure
  - Red = fault
  - Obsidian = environment

IMPORTANT:
  Geometry and composition intentionally remain unchanged.
  State changes only affect visual behavior.
============================================================
*/

const STATE_CONFIG = {
  idle: {
    color: "#3B82F6",
    glow: "rgba(59,130,246,0.30)",
    speed: 4.5,
    movement: 1,
    textColor: "#EAF2FF",
  },

  listening: {
    color: "#93C5FD",
    glow: "rgba(147,197,253,0.42)",
    speed: 1.8,
    movement: 1.8,
    textColor: "#F3F7FF",
  },

  thinking: {
    color: "#60A5FA",
    glow: "rgba(96,165,250,0.40)",
    speed: 2.4,
    movement: 2.2,
    textColor: "#EDF5FF",
  },

  executing: {
    color: "#3B82F6",
    glow: "rgba(59,130,246,0.50)",
    speed: 1.25,
    movement: 2.8,
    textColor: "#F2F7FF",
  },

  speaking: {
    color: "#60A5FA",
    glow: "rgba(96,165,250,0.55)",
    speed: 0.9,
    movement: 3,
    textColor: "#F7FAFF",
  },

  notifying: {
    color: "#93C5FD",
    glow: "rgba(147,197,253,0.58)",
    speed: 0.65,
    movement: 3.4,
    textColor: "#F7FAFF",
  },

  muted: {
    color: "#64748B",
    glow: "rgba(100,116,139,0.16)",
    speed: 7,
    movement: 0.35,
    textColor: "#CBD5E1",
  },

  error: {
    color: "#F87171",
    glow: "rgba(248,113,113,0.52)",
    speed: 0.55,
    movement: 4,
    textColor: "#FFF1F2",
  },
};

const STATE_LABELS = {
  idle: "IDLE",
  listening: "LISTENING",
  thinking: "PROCESSING",
  executing: "EXECUTING",
  speaking: "SPEAKING",
  notifying: "ALERT",
  muted: "MUTED",
  error: "FAULT",
};

export default function ZoeOrb({
  state = "idle",
  label = "Z.O.E",
}) {
  const cfg = STATE_CONFIG[state] || STATE_CONFIG.idle;

  return (
    <div
      className={`zoe-loader zoe-${state}`}
      style={{
        "--zoe-color": cfg.color,
        "--zoe-glow": cfg.glow,
        "--zoe-speed": `${cfg.speed}s`,
        "--zoe-movement": cfg.movement,
        "--zoe-text-color": cfg.textColor,
      }}
    >
      <style>{`

        /* ====================================================
           ZOE CORE MOTION
        ==================================================== */

        @keyframes zoe-outer-breathe {
          0%,
          100% {
            transform: scale(0.985);
            opacity: 0.68;
          }

          50% {
            transform:
              scale(
                calc(1 + (0.006 * var(--zoe-movement)))
              );

            opacity: 1;
          }
        }

        @keyframes zoe-inner-breathe {
          0%,
          100% {
            transform: scale(0.975);
            opacity: 0.64;
          }

          35% {
            transform:
              scale(
                calc(1 + (0.008 * var(--zoe-movement)))
              );

            opacity: 0.95;
          }

          70% {
            transform:
              scale(
                calc(1 - (0.003 * var(--zoe-movement)))
              );

            opacity: 0.78;
          }
        }

        @keyframes zoe-speaking-wave {
          0% {
            transform: scale(0.98);
          }

          25% {
            transform:
              scale(
                calc(1 + (0.01 * var(--zoe-movement)))
              );
          }

          50% {
            transform: scale(0.995);
          }

          75% {
            transform:
              scale(
                calc(1 - (0.006 * var(--zoe-movement)))
              );
          }

          100% {
            transform: scale(0.98);
          }
        }

        @keyframes zoe-glow {
          0%,
          100% {
            opacity: 0.28;
            transform: scale(0.96);
          }

          50% {
            opacity: 0.72;
            transform: scale(1.04);
          }
        }

        @keyframes zoe-label-pulse {
          0%,
          100% {
            opacity: 0.68;
          }

          50% {
            opacity: 1;
          }
        }


        /* ====================================================
           THINKING
           Subtle processing movement.
        ==================================================== */

        @keyframes zoe-thinking {
          0% {
            transform: scale(0.985) rotate(0deg);
            opacity: 0.72;
          }

          25% {
            transform: scale(1.008) rotate(0.35deg);
            opacity: 0.92;
          }

          50% {
            transform: scale(0.992) rotate(-0.25deg);
            opacity: 1;
          }

          75% {
            transform: scale(1.004) rotate(0.2deg);
            opacity: 0.88;
          }

          100% {
            transform: scale(0.985) rotate(0deg);
            opacity: 0.72;
          }
        }


        /* ====================================================
           EXECUTING
           Faster, more energetic system state.
        ==================================================== */

        @keyframes zoe-executing {
          0%,
          100% {
            transform: scale(0.985);
            opacity: 0.76;
          }

          20% {
            transform: scale(1.018);
            opacity: 1;
          }

          42% {
            transform: scale(0.995);
            opacity: 0.84;
          }

          65% {
            transform: scale(1.012);
            opacity: 0.96;
          }

          82% {
            transform: scale(0.99);
            opacity: 0.80;
          }
        }


        /* ====================================================
           NOTIFYING
           Attention pulse without looking like an error.
        ==================================================== */

        @keyframes zoe-notifying {
          0%,
          100% {
            transform: scale(0.985);
            opacity: 0.70;
          }

          18% {
            transform: scale(1.025);
            opacity: 1;
          }

          32% {
            transform: scale(0.997);
            opacity: 0.88;
          }

          50% {
            transform: scale(1.012);
            opacity: 0.96;
          }

          68% {
            transform: scale(0.992);
            opacity: 0.82;
          }
        }


        /* ====================================================
           ERROR
           Controlled fault instability.
        ==================================================== */

        @keyframes zoe-error {
          0%,
          100% {
            transform: scale(0.975);
            opacity: 0.62;
          }

          12% {
            transform: scale(1.02);
            opacity: 1;
          }

          18% {
            transform: scale(0.99);
            opacity: 0.72;
          }

          27% {
            transform: scale(1.028);
            opacity: 0.95;
          }

          35% {
            transform: scale(0.982);
            opacity: 0.68;
          }

          52% {
            transform: scale(1.012);
            opacity: 0.92;
          }

          70% {
            transform: scale(0.992);
            opacity: 0.76;
          }
        }


        /* ====================================================
           MUTED
        ==================================================== */

        @keyframes zoe-muted {
          0%,
          100% {
            transform: scale(0.988);
            opacity: 0.42;
          }

          50% {
            transform: scale(0.998);
            opacity: 0.58;
          }
        }


        /* ====================================================
           CONTAINER
        ==================================================== */

        .zoe-loader {
          width: 100%;
          min-height: 360px;

          display: flex;
          align-items: center;
          justify-content: center;

          font-family:
            "Orbitron",
            "SFMono-Regular",
            Consolas,
            monospace;

          color: var(--zoe-text-color);
        }


        /* ====================================================
           ORB
        ==================================================== */

        .zoe-orb {
          position: relative;

          width: 300px;
          height: 300px;

          display: flex;
          align-items: center;
          justify-content: center;
        }


        /* ====================================================
           AMBIENT ENERGY
        ==================================================== */

        .zoe-ambient {
          position: absolute;

          width: 185px;
          height: 185px;

          border-radius: 50%;

          background:
            radial-gradient(
              circle,
              var(--zoe-glow) 0%,
              rgba(59, 130, 246, 0.08) 32%,
              transparent 70%
            );

          filter: blur(24px);

          animation:
            zoe-glow
            4s
            ease-in-out
            infinite;

          pointer-events: none;
        }


        /* ====================================================
           RINGS
        ==================================================== */

        .zoe-ring {
          position: absolute;

          inset: 0;

          width: 100%;
          height: 100%;

          transform-origin: center;

          overflow: visible;
        }

        .zoe-outer {
          animation:
            zoe-outer-breathe
            var(--zoe-speed)
            ease-in-out
            infinite;

          filter:
            drop-shadow(
              0 0 4px var(--zoe-glow)
            );
        }

        .zoe-inner {
          animation:
            zoe-inner-breathe
            calc(var(--zoe-speed) * 1.17)
            ease-in-out
            infinite;

          filter:
            drop-shadow(
              0 0 2px
              rgba(229, 231, 235, 0.18)
            );
        }


        /* ====================================================
           CENTER
        ==================================================== */

        .zoe-center {
          position: absolute;

          inset: 0;

          display: flex;
          align-items: center;
          justify-content: center;

          flex-direction: column;

          gap: 6px;

          pointer-events: none;

          z-index: 5;
        }


        /* ====================================================
           NAME
        ==================================================== */

        .zoe-name {
          color: var(--zoe-text-color);

          font-size: 25px;
          font-weight: 500;

          letter-spacing: 5px;

          text-shadow:
            0 0 6px var(--zoe-glow),
            0 0 18px rgba(96, 165, 250, 0.14);

          animation:
            zoe-label-pulse
            3s
            ease-in-out
            infinite;
        }


        /* ====================================================
           STATE
        ==================================================== */

        .zoe-state {
          color: var(--zoe-color);

          font-size: 7px;
          font-weight: 500;

          letter-spacing: 3px;

          opacity: 0.42;

          text-shadow:
            0 0 5px var(--zoe-glow);
        }


        /* ====================================================
           IDLE
        ==================================================== */

        .zoe-idle .zoe-outer {
          animation-duration: 5s;
        }

        .zoe-idle .zoe-inner {
          animation-duration: 5.8s;
        }

        .zoe-idle .zoe-ambient {
          opacity: 0.22;
          animation-duration: 5s;
        }


        /* ====================================================
           LISTENING
        ==================================================== */

        .zoe-listening .zoe-outer {
          animation-duration: 2.2s;

          filter:
            drop-shadow(
              0 0 5px var(--zoe-glow)
            );
        }

        .zoe-listening .zoe-inner {
          animation-duration: 1.8s;
        }

        .zoe-listening .zoe-ambient {
          animation-duration: 2.2s;
          opacity: 0.48;
        }


        /* ====================================================
           THINKING
        ==================================================== */

        .zoe-thinking .zoe-outer {
          animation:
            zoe-thinking
            2.4s
            ease-in-out
            infinite;

          filter:
            drop-shadow(
              0 0 5px var(--zoe-glow)
            );
        }

        .zoe-thinking .zoe-inner {
          animation:
            zoe-thinking
            1.85s
            ease-in-out
            infinite reverse;
        }

        .zoe-thinking .zoe-ambient {
          animation-duration: 2.6s;
          opacity: 0.46;
        }

        .zoe-thinking .zoe-name {
          animation-duration: 2.1s;
        }


        /* ====================================================
           EXECUTING
        ==================================================== */

        .zoe-executing .zoe-outer {
          animation:
            zoe-executing
            1.25s
            ease-in-out
            infinite;

          filter:
            drop-shadow(
              0 0 5px var(--zoe-glow)
            )
            drop-shadow(
              0 0 12px
              rgba(59,130,246,0.18)
            );
        }

        .zoe-executing .zoe-inner {
          animation:
            zoe-executing
            0.92s
            ease-in-out
            infinite reverse;
        }

        .zoe-executing .zoe-ambient {
          animation-duration: 1.15s;
          opacity: 0.64;
        }

        .zoe-executing .zoe-name {
          animation-duration: 1.5s;
        }


        /* ====================================================
           SPEAKING
        ==================================================== */

        .zoe-speaking .zoe-outer {
          animation:
            zoe-speaking-wave
            1s
            ease-in-out
            infinite;

          filter:
            drop-shadow(
              0 0 5px var(--zoe-glow)
            )
            drop-shadow(
              0 0 14px
              rgba(96, 165, 250, 0.20)
            );
        }

        .zoe-speaking .zoe-inner {
          animation:
            zoe-inner-breathe
            0.72s
            ease-in-out
            infinite;
        }

        .zoe-speaking .zoe-ambient {
          animation-duration: 1.2s;
          opacity: 0.70;
        }

        .zoe-speaking .zoe-name {
          animation-duration: 1.1s;
        }


        /* ====================================================
           NOTIFYING
        ==================================================== */

        .zoe-notifying .zoe-outer {
          animation:
            zoe-notifying
            0.75s
            ease-in-out
            infinite;

          filter:
            drop-shadow(
              0 0 6px var(--zoe-glow)
            )
            drop-shadow(
              0 0 15px
              rgba(147,197,253,0.18)
            );
        }

        .zoe-notifying .zoe-inner {
          animation:
            zoe-notifying
            0.95s
            ease-in-out
            infinite reverse;
        }

        .zoe-notifying .zoe-ambient {
          animation-duration: 0.8s;
          opacity: 0.72;
        }

        .zoe-notifying .zoe-name {
          animation-duration: 0.8s;
        }


        /* ====================================================
           MUTED
        ==================================================== */

        .zoe-muted .zoe-outer {
          animation:
            zoe-muted
            7s
            ease-in-out
            infinite;

          filter:
            drop-shadow(
              0 0 2px var(--zoe-glow)
            );
        }

        .zoe-muted .zoe-inner {
          animation:
            zoe-muted
            8s
            ease-in-out
            infinite reverse;

          opacity: 0.55;
        }

        .zoe-muted .zoe-ambient {
          animation-duration: 8s;
          opacity: 0.12;
        }

        .zoe-muted .zoe-name {
          opacity: 0.55;
          text-shadow:
            0 0 5px var(--zoe-glow);
        }

        .zoe-muted .zoe-state {
          opacity: 0.32;
        }


        /* ====================================================
           ERROR
        ==================================================== */

        .zoe-error .zoe-outer {
          animation:
            zoe-error
            0.55s
            ease-in-out
            infinite;

          filter:
            drop-shadow(
              0 0 5px var(--zoe-glow)
            )
            drop-shadow(
              0 0 14px
              rgba(248,113,113,0.18)
            );
        }

        .zoe-error .zoe-inner {
          animation:
            zoe-error
            0.7s
            ease-in-out
            infinite reverse;
        }

        .zoe-error .zoe-ambient {
          animation:
            zoe-glow
            0.7s
            ease-in-out
            infinite;

          opacity: 0.75;
        }

        .zoe-error .zoe-name {
          animation:
            zoe-label-pulse
            0.55s
            ease-in-out
            infinite;
        }


        /* ====================================================
           ACCESSIBILITY
        ==================================================== */

        @media (prefers-reduced-motion: reduce) {
          .zoe-loader *,
          .zoe-loader *::before,
          .zoe-loader *::after {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
          }
        }


        /* ====================================================
           RESPONSIVE
        ==================================================== */

        @media (max-width: 600px) {
          .zoe-loader {
            min-height: 300px;
          }

          .zoe-orb {
            width: 250px;
            height: 250px;
          }

          .zoe-name {
            font-size: 21px;
            letter-spacing: 3.5px;
          }
        }

      `}</style>

      <div className="zoe-orb">

        {/* Ambient energy */}
        <div className="zoe-ambient" />

        {/* Outer electric-blue ring */}
        <svg
          className="zoe-ring zoe-outer"
          viewBox="0 0 250 250"
          aria-hidden="true"
        >
          <circle
            cx="125"
            cy="125"
            r="108"
            fill="none"
            stroke="var(--zoe-color)"
            strokeWidth="3"
          />
        </svg>

        {/* Inner silver ring */}
        <svg
          className="zoe-ring zoe-inner"
          viewBox="0 0 250 250"
          aria-hidden="true"
        >
          <circle
            cx="125"
            cy="125"
            r="99"
            fill="none"
            stroke="rgba(229,231,235,0.90)"
            strokeWidth="2.2"
          />
        </svg>

        {/* Center identity */}
        <div className="zoe-center">
          <div className="zoe-name">
            Z.O.E
          </div>

          <div className="zoe-state">
            {STATE_LABELS[state] || STATE_LABELS.idle}
          </div>
        </div>

      </div>
    </div>
  );
}