
import { useEffect, useRef } from "react";

/* ============================================================
   ZOE GLOBAL UI CLICK SOUND
   ------------------------------------------------------------
   Plays /sounds/click.mp3 whenever an interactive UI element
   is clicked anywhere inside the application.

   Covered:
   - Buttons
   - Links
   - Checkboxes
   - Radio buttons
   - Range sliders
   - Selects
   - File/color inputs
   - Elements with role="button"
   - Elements with a valid tabindex

   Excluded:
   - Text inputs
   - Textareas
   - Disabled controls
   - Elements explicitly marked data-no-click-sound
============================================================ */

export default function GlobalClickSound() {

    const audioRef = useRef(null);


    /* ========================================================
       INITIALIZE
    ======================================================== */

    useEffect(() => {

        const audio =
            new Audio("/sounds/click.mp3");


        /*
         * Preload the sound so the first click has
         * as little latency as possible.
         */
        audio.preload = "auto";


        /*
         * Keep the click subtle.
         *
         * Adjust between 0.10 - 0.30 depending on
         * the actual loudness of your click.mp3.
         */
        audio.volume = 0.22;


        audioRef.current = audio;


        /* ====================================================
           CLICK HANDLER
        ==================================================== */

        const handleClick = (event) => {

            /*
             * Make sure the click originated from an
             * actual DOM Element.
             */
            const target =
                event.target instanceof Element
                    ? event.target
                    : null;


            if (!target) {
                return;
            }


            /* =================================================
               FIND INTERACTIVE ELEMENT
            ================================================= */

            const interactive =
                target.closest(
                    [
                        "button",
                        "a",
                        "input",
                        "select",
                        "textarea",
                        "[role='button']",
                        "[role='link']",
                        "[role='checkbox']",
                        "[role='radio']",
                        "[role='switch']",
                        "[role='tab']",
                        "[role='menuitem']",
                        "[role='option']",
                        "[role='slider']",
                        "[tabindex]:not([tabindex='-1'])",
                    ].join(",")
                );


            /*
             * Click wasn't on an interactive element.
             */
            if (!interactive) {
                return;
            }


            /* =================================================
               EXPLICIT OPT-OUT
            ================================================= */

            /*
             * Any element can disable the click sound:
             *
             * <button data-no-click-sound>
             */
            if (
                interactive.hasAttribute(
                    "data-no-click-sound"
                )
            ) {
                return;
            }


            /*
             * Also allow a parent container to disable
             * the sound for all of its children.
             */
            if (
                interactive.closest(
                    "[data-no-click-sound]"
                )
            ) {
                return;
            }


            /* =================================================
               DISABLED CONTROLS
            ================================================= */

            if (
                interactive.hasAttribute("disabled") ||
                interactive.getAttribute("aria-disabled") === "true"
            ) {
                return;
            }


            /* =================================================
               INPUT HANDLING
            ================================================= */

            const tag =
                interactive.tagName.toLowerCase();


            const inputType =
                interactive instanceof HTMLInputElement
                    ? interactive.type.toLowerCase()
                    : "";


            /*
             * Text-entry controls should NOT make a click
             * sound when the user clicks into them.
             */
            if (
                tag === "textarea"
            ) {
                return;
            }


            if (
                tag === "input" &&
                [
                    "text",
                    "search",
                    "email",
                    "password",
                    "tel",
                    "url",
                    "number",
                    "date",
                    "datetime-local",
                    "month",
                    "time",
                    "week",
                ].includes(inputType)
            ) {
                return;
            }


            /* =================================================
               PLAY SOUND
            ================================================= */

            const clickSound =
                audioRef.current;


            if (!clickSound) {
                return;
            }


            /*
             * Restart immediately.
             *
             * This makes rapid clicking behave naturally:
             *
             * click → sound
             * click → restart
             * click → restart
             */
            try {

                clickSound.currentTime = 0;

            } catch {
                /*
                 * Some browsers may throw if the audio
                 * isn't seekable yet. Playback can still
                 * be attempted below.
                 */
            }


            const playPromise =
                clickSound.play();


            /*
             * Modern browsers return a Promise from play().
             */
            if (
                playPromise &&
                typeof playPromise.catch === "function"
            ) {

                playPromise.catch(() => {
                    /*
                     * Ignore browser autoplay/audio-policy
                     * errors. Never let UI interaction fail
                     * because the sound couldn't play.
                     */
                });

            }

        };


        /* ====================================================
           GLOBAL EVENT
        ==================================================== */

        /*
         * Capture phase:
         *
         * true
         *
         * This allows the global sound to detect clicks even
         * if a React component later stops propagation.
         */
        document.addEventListener(
            "click",
            handleClick,
            true
        );


        /* ====================================================
           CLEANUP
        ==================================================== */

        return () => {

            document.removeEventListener(
                "click",
                handleClick,
                true
            );


            if (audioRef.current) {

                audioRef.current.pause();

                audioRef.current.currentTime = 0;

                audioRef.current = null;
            }

        };

    }, []);


    /* ========================================================
       NO VISUAL OUTPUT
    ======================================================== */

    return null;
}
