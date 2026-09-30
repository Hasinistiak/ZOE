import React, {
    useEffect,
    useRef,
    useState,
} from "react";

import {
    WebviewWindow,
} from "@tauri-apps/api/webviewWindow";

import ZoeOrb from "./widgets/ZoeOrb";
import ProcessingHUD from "./widgets/ProcessingHUD";

import "./App.css";

import ChatPage from "./ChatPage";

import {
    SurfaceStage,
    NotificationStack,
    ActivityStream,
} from "./widgets/ZoeSystems";

import GlobalClickSound from "./widgets/GlobalClickSound";


const API_BASE = import.meta.env.DEV
    ? ""
    : "http://127.0.0.1:8000";


// ============================================================
// STATE METADATA
// ============================================================

const STATE_META = {

    idle: {
        label: "STANDING BY",
    },

    listening: {
        label: "LISTENING",
    },

    thinking: {
        label: "PROCESSING",
    },

    executing: {
        label: "EXECUTING",
    },

    speaking: {
        label: "RESPONDING",
    },

    notifying: {
        label: "ALERTING",
    },

    muted: {
        label: "MIC MUTED",
    },

    error: {
        label: "FAULT",
    },

};


// ============================================================
// CONSTANTS
// ============================================================

const STATE_POLL_INTERVAL = 150;


// ============================================================
// MICROPHONE ICONS
// ============================================================

function MicrophoneIcon() {

    return (
        <svg
            className="zoe-mic-svg"
            viewBox="0 0 24 24"
            aria-hidden="true"
        >
            <rect
                x="8"
                y="3"
                width="8"
                height="12"
                rx="4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
            />

            <path
                d="M5.5 11.5C5.5 15.09 8.41 18 12 18s6.5-2.91 6.5-6.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />

            <path
                d="M12 18v3"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />

            <path
                d="M9 21h6"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />
        </svg>
    );

}


function MicrophoneOffIcon() {

    return (
        <svg
            className="zoe-mic-svg"
            viewBox="0 0 24 24"
            aria-hidden="true"
        >
            <rect
                x="8"
                y="3"
                width="8"
                height="12"
                rx="4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
            />

            <path
                d="M5.5 11.5C5.5 15.09 8.41 18 12 18"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />

            <path
                d="M18.5 11.5C18.5 13.1 17.92 14.56 16.95 15.67"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />

            <path
                d="M12 18v3"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />

            <path
                d="M9 21h6"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.7"
                strokeLinecap="round"
            />

            <path
                d="M4 4l16 16"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.9"
                strokeLinecap="round"
            />
        </svg>
    );

}


// ============================================================
// ZOE DASHBOARD
// ============================================================

export default function ZoeDashboard() {

    // ========================================================
    // CORE STATE
    // ========================================================

    const [
        zoeState,
        setZoeState,
    ] = useState("idle");


    const [
        response,
        setResponse,
    ] = useState("");


    const [
        online,
        setOnline,
    ] = useState(false);


    const [
        muted,
        setMuted,
    ] = useState(false);


    // ========================================================
    // CINEMATIC PROCESSING STATE
    // ========================================================

    const [
        processingState,
        setProcessingState,
    ] = useState("PROCESSING");


    const [
        processingOperation,
        setProcessingOperation,
    ] = useState("");


    const [
        processingDetail,
        setProcessingDetail,
    ] = useState("");


    // ========================================================
    // BACKGROUND WORK STATE
    // ========================================================

    const [
        backgroundWork,
        setBackgroundWork,
    ] = useState(null);


    // ========================================================
    // ACTIVE PAGE
    // ========================================================

    const [
        activePage,
        setActivePage,
    ] = useState(null);


    // ========================================================
    // SYSTEMS STATE
    // ========================================================

    const [
        surface,
        setSurface,
    ] = useState(null);


    const [
        notifications,
        setNotifications,
    ] = useState([]);


    const [
        activity,
        setActivity,
    ] = useState([]);


    // ========================================================
    // STT STATE
    // ========================================================

    const [
        sttText,
        setSttText,
    ] = useState("");


    const [
        finalSttText,
        setFinalSttText,
    ] = useState("");


    const [
        userSpeaking,
        setUserSpeaking,
    ] = useState(false);


    const [
        sttDisplayActive,
        setSttDisplayActive,
    ] = useState(false);


    // ========================================================
    // RESPONSE DISPLAY STATE
    // ========================================================

    const [
        displayedResponse,
        setDisplayedResponse,
    ] = useState("");


    // ========================================================
    // REFS
    // ========================================================

    const stateRequestId =
        useRef(0);


    const sttDisplayActiveRef =
        useRef(false);


    const responseSpokenRef =
        useRef(false);


    const sttClearTimer =
        useRef(null);


    const lastFinalTextRef =
        useRef("");


    // ========================================================
    // RESPONSE ANIMATION REFS
    // ========================================================

    const responseAnimationRef =
        useRef(null);


    const responseIndexRef =
        useRef(0);


    const previousResponseRef =
        useRef("");


    // ========================================================
    // TAURI EXTERNAL WINDOW REFS
    // ========================================================

    const openedUrlsRef =
        useRef(
            new Set()
        );


    const externalWindowCounterRef =
        useRef(0);


    // ========================================================
    // OPEN URL IN NEW TAURI WINDOW
    // ========================================================

    async function openExternalUrl(
        url
    ) {

        const cleanUrl =
            typeof url === "string"
                ? url.trim()
                : "";


        if (
            !cleanUrl
        ) {

            return;

        }


        let parsedUrl;


        try {

            parsedUrl =
                new URL(
                    cleanUrl
                );

        } catch (error) {

            console.error(
                "[ZOE] Invalid URL:",
                cleanUrl,
                error
            );

            return;

        }


        if (
            parsedUrl.protocol !== "http:" &&
            parsedUrl.protocol !== "https:"
        ) {

            console.error(
                "[ZOE] Refusing unsupported URL scheme:",
                parsedUrl.protocol
            );

            return;

        }


        if (
            openedUrlsRef.current.has(
                cleanUrl
            )
        ) {

            return;

        }


        openedUrlsRef.current.add(
            cleanUrl
        );


        externalWindowCounterRef.current += 1;


        const windowNumber =
            externalWindowCounterRef.current;


        const label =
            `zoe-external-${windowNumber}`;


        let hostname =
            parsedUrl.hostname;


        try {

            hostname =
                decodeURIComponent(
                    hostname
                );

        } catch {

            // Keep original hostname.

        }


        try {

            const externalWindow =
                new WebviewWindow(
                    label,
                    {

                        url:
                            cleanUrl,

                        title:
                            `ZOE · ${hostname}`,

                        width:
                            1200,

                        height:
                            800,

                        resizable:
                            true,

                        center:
                            true,

                        focus:
                            true,

                        visible:
                            true,

                    }
                );


            await new Promise(
                (resolve) => {

                    externalWindow.once(
                        "tauri://created",
                        () => {

                            console.log(
                                `[ZOE] External window created: ${label}`,
                                cleanUrl
                            );

                            resolve();

                        }
                    );


                    externalWindow.once(
                        "tauri://error",
                        (event) => {

                            console.error(
                                `[ZOE] Failed to create external window: ${label}`,
                                event
                            );


                            openedUrlsRef.current.delete(
                                cleanUrl
                            );


                            resolve();

                        }
                    );

                }
            );


        } catch (error) {

            console.error(
                "[ZOE] Failed to open Tauri external window:",
                error
            );


            openedUrlsRef.current.delete(
                cleanUrl
            );

        }

    }


    // ========================================================
    // RESPONSE TYPING / GENERATION EFFECT
    // ========================================================

    useEffect(() => {

        const target =
            typeof response === "string"
                ? response
                : "";


        if (
            responseAnimationRef.current
        ) {

            clearTimeout(
                responseAnimationRef.current
            );

            responseAnimationRef.current =
                null;

        }


        if (
            !target
        ) {

            responseIndexRef.current =
                0;

            previousResponseRef.current =
                "";

            setDisplayedResponse(
                ""
            );

            return;

        }


        const previous =
            previousResponseRef.current;


        const isContinuation =
            Boolean(
                previous &&
                target.startsWith(
                    previous
                )
            );


        if (
            !isContinuation
        ) {

            responseIndexRef.current =
                0;

            setDisplayedResponse(
                ""
            );

        } else {

            responseIndexRef.current =
                Math.min(
                    responseIndexRef.current,
                    target.length
                );

        }


        previousResponseRef.current =
            target;


        function revealNext() {

            const currentTarget =
                previousResponseRef.current;


            const currentIndex =
                responseIndexRef.current;


            if (
                currentIndex >=
                currentTarget.length
            ) {

                setDisplayedResponse(
                    currentTarget
                );


                responseAnimationRef.current =
                    null;


                return;

            }


            const remaining =
                currentTarget.length -
                currentIndex;


            let chunkSize =
                1;


            if (
                remaining > 500
            ) {

                chunkSize =
                    5;

            } else if (
                remaining > 300
            ) {

                chunkSize =
                    4;

            } else if (
                remaining > 180
            ) {

                chunkSize =
                    3;

            } else if (
                remaining > 80
            ) {

                chunkSize =
                    2;

            }


            const nextIndex =
                Math.min(
                    currentIndex +
                    chunkSize,
                    currentTarget.length
                );


            responseIndexRef.current =
                nextIndex;


            setDisplayedResponse(
                currentTarget.slice(
                    0,
                    nextIndex
                )
            );


            const lastChar =
                currentTarget[
                    nextIndex - 1
                ];


            let delay =
                24;


            if (
                lastChar === "." ||
                lastChar === "!" ||
                lastChar === "?"
            ) {

                delay =
                    90;

            } else if (
                lastChar === "," ||
                lastChar === ":" ||
                lastChar === ";"
            ) {

                delay =
                    50;

            } else if (
                lastChar === "\n"
            ) {

                delay =
                    70;

            } else if (
                lastChar === " "
            ) {

                delay =
                    18;

            }


            responseAnimationRef.current =
                setTimeout(
                    revealNext,
                    delay
                );

        }


        revealNext();


        return () => {

            if (
                responseAnimationRef.current
            ) {

                clearTimeout(
                    responseAnimationRef.current
                );

                responseAnimationRef.current =
                    null;

            }

        };

    }, [response]);


    // ========================================================
    // BACKEND STATE POLLING
    // ========================================================

    useEffect(() => {

        let mounted =
            true;


        async function updateState() {

            const requestId =
                ++stateRequestId.current;


            try {

                const res =
                    await fetch(
                        `${API_BASE}/api/state`,
                        {
                            method: "GET",
                            cache: "no-store",
                        }
                    );


                if (
                    !res.ok
                ) {

                    throw new Error(
                        `HTTP ${res.status}`
                    );

                }


                const data =
                    await res.json();


                if (
                    !mounted ||
                    requestId !==
                    stateRequestId.current
                ) {

                    return;

                }


                setOnline(
                    data.online === true
                );


                const currentZoeState =
                    getBackendState(
                        data
                    );


                setZoeState(
                    currentZoeState
                );


                const processing =
                    data.processing === true;


                const currentBackgroundWork =
                    resolveBackgroundWork(
                        data
                    );


                setBackgroundWork(
                    currentBackgroundWork
                );


                const currentProcessingState =
                    resolveProcessingMode(
                        data
                    );


                const currentOperation =
                    resolveProcessingOperation(
                        data
                    );


                const currentDetail =
                    resolveProcessingDetail(
                        data
                    );


                setProcessingState(
                    currentProcessingState
                );


                setProcessingOperation(
                    currentOperation
                );


                setProcessingDetail(
                    currentDetail
                );


                setMuted(
                    data.muted === true
                );


                setSurface(
                    data.surface ||
                    null
                );


                if (
                    Array.isArray(
                        data.notifications
                    )
                ) {

                    setNotifications(
                        data.notifications
                    );

                }


                if (
                    Array.isArray(
                        data.activity
                    )
                ) {

                    setActivity(
                        data.activity
                    );

                }


                if (
                    typeof data.response ===
                    "string"
                ) {

                    setResponse(
                        data.response
                    );

                }


                const backendUrl =
                    typeof data.url === "string"
                        ? data.url.trim()
                        : "";


                if (
                    backendUrl
                ) {

                    await openExternalUrl(
                        backendUrl
                    );

                }


                const stt =
                    data.stt ||
                    {};


                const liveText =
                    typeof stt.text ===
                        "string"
                        ? stt.text.trim()
                        : "";


                const finalText =
                    typeof stt.final_text ===
                        "string"
                        ? stt.final_text.trim()
                        : "";


                const speaking =
                    stt.speaking === true;


                const newFinalText =
                    Boolean(
                        finalText &&
                        finalText !==
                        lastFinalTextRef.current
                    );


                if (
                    speaking ||
                    newFinalText ||
                    processing
                ) {

                    if (
                        sttClearTimer.current
                    ) {

                        clearTimeout(
                            sttClearTimer.current
                        );

                        sttClearTimer.current =
                            null;

                    }


                    sttDisplayActiveRef.current =
                        true;


                    setSttDisplayActive(
                        true
                    );

                }


                if (
                    newFinalText
                ) {

                    lastFinalTextRef.current =
                        finalText;

                }


                if (
                    currentZoeState ===
                    "speaking"
                ) {

                    responseSpokenRef.current =
                        true;

                }


                if (
                    currentZoeState ===
                    "idle" &&
                    sttDisplayActiveRef.current &&
                    !processing &&
                    !speaking
                ) {

                    if (
                        responseSpokenRef.current
                    ) {

                        sttDisplayActiveRef.current =
                            false;


                        responseSpokenRef.current =
                            false;


                        setSttDisplayActive(
                            false
                        );

                    } else if (
                        !sttClearTimer.current
                    ) {

                        sttClearTimer.current =
                            setTimeout(
                                () => {

                                    sttDisplayActiveRef.current =
                                        false;


                                    responseSpokenRef.current =
                                        false;


                                    setSttDisplayActive(
                                        false
                                    );


                                    sttClearTimer.current =
                                        null;

                                },
                                1200
                            );

                    }

                }


                setUserSpeaking(
                    speaking
                );


                setSttText(
                    liveText
                );


                setFinalSttText(
                    finalText
                );


            } catch (error) {

                if (
                    !mounted ||
                    requestId !==
                    stateRequestId.current
                ) {

                    return;

                }


                console.error(
                    "[ZOE] Backend unavailable:",
                    error
                );


                setOnline(
                    false
                );


                setZoeState(
                    "error"
                );


                setProcessingState(
                    "ERROR"
                );


                setProcessingOperation(
                    "BACKEND CONNECTION"
                );


                setProcessingDetail(
                    "Unable to communicate with ZOE runtime."
                );


                setBackgroundWork(
                    null
                );


                setUserSpeaking(
                    false
                );

            }

        }


        updateState();


        const interval =
            setInterval(
                updateState,
                STATE_POLL_INTERVAL
            );


        return () => {

            mounted =
                false;


            clearInterval(
                interval
            );


            if (
                sttClearTimer.current
            ) {

                clearTimeout(
                    sttClearTimer.current
                );


                sttClearTimer.current =
                    null;

            }


            if (
                responseAnimationRef.current
            ) {

                clearTimeout(
                    responseAnimationRef.current
                );


                responseAnimationRef.current =
                    null;

            }

        };

    }, []);


    // ========================================================
    // MUTE / UNMUTE
    // ========================================================

    async function toggleMute() {

        try {

            const endpoint =
                muted
                    ? `${API_BASE}/api/microphone/unmute`
                    : `${API_BASE}/api/microphone/mute`;


            const res =
                await fetch(
                    endpoint,
                    {
                        method: "POST",
                    }
                );


            const data =
                await res.json();


            if (
                !res.ok ||
                !data.success
            ) {

                throw new Error(
                    data.error ||
                    `HTTP ${res.status}`
                );

            }


            setMuted(
                data.muted === true
            );


            if (
                data.muted === true
            ) {

                setUserSpeaking(
                    false
                );


                setSttText(
                    ""
                );


                setFinalSttText(
                    ""
                );


                setSttDisplayActive(
                    false
                );


                sttDisplayActiveRef.current =
                    false;


                responseSpokenRef.current =
                    false;


                if (
                    sttClearTimer.current
                ) {

                    clearTimeout(
                        sttClearTimer.current
                    );


                    sttClearTimer.current =
                        null;

                }

            }


        } catch (error) {

            console.error(
                "[ZOE] Failed to toggle microphone:",
                error
            );

        }

    }


    // ========================================================
    // PAGE NAVIGATION
    // ========================================================

    function openPage(
        page
    ) {

        setActivePage(
            page
        );

    }


    function closePage() {

        setActivePage(
            null
        );

    }


    // ========================================================
    // SHOW FINAL TRANSCRIPT
    // ========================================================

    const showingFinal =
        !userSpeaking &&
        sttDisplayActive &&
        Boolean(
            finalSttText &&
            finalSttText.trim()
        );


    // ========================================================
    // DISPLAYED TRANSCRIPT
    // ========================================================

    let displayedStt =
        "";


    if (
        userSpeaking
    ) {

        displayedStt =
            sttText ||
            "";

    } else if (
        showingFinal
    ) {

        displayedStt =
            finalSttText ||
            "";

    } else if (
        sttDisplayActive
    ) {

        displayedStt =
            sttText ||
            "";

    }


    displayedStt =
        displayedStt.trim();


    // ========================================================
    // ORB STATE
    // ========================================================

    const orbState =
        zoeState;


    // ========================================================
    // FOREGROUND PROCESSING
    // ========================================================

    const foregroundProcessing =
        online &&
        (
            zoeState === "thinking" ||
            zoeState === "executing" ||
            processingState === "AGENT" ||
            processingState === "RESEARCH" ||
            processingState === "TOOL"
        );


    // ========================================================
    // BACKGROUND PROCESSING
    // ========================================================

    const backgroundProcessing =
        online &&
        backgroundWork?.active === true;


    // ========================================================
    // GLOBAL PROCESSING ACTIVITY
    // ========================================================

    const processingActive =
        foregroundProcessing ||
        backgroundProcessing;


    // ========================================================
    // EFFECTIVE HUD STATE
    // ========================================================

    const effectiveProcessingState =
        backgroundProcessing
            ? "AGENT"
            : processingState;


    // ========================================================
    // BACKGROUND AGENT NAME
    // ========================================================

    const backgroundAgentName =
        backgroundWork?.name ||
        backgroundWork?.agentName ||
        backgroundWork?.agent_name ||
        backgroundWork?.agents?.[0]?.name ||
        "";


    // ========================================================
    // EFFECTIVE HUD OPERATION
    // ========================================================

    const effectiveProcessingOperation =
        backgroundProcessing
            ? (
                backgroundWork.operation ||
                backgroundWork.currentOperation ||
                backgroundWork.current_operation ||
                backgroundWork.agents?.[0]?.detail ||
                backgroundAgentName ||
                "BACKGROUND WORK"
            )
            : processingOperation;


    // ========================================================
    // EFFECTIVE HUD DETAIL
    // ========================================================

    const effectiveProcessingDetail =
        backgroundProcessing
            ? (
                backgroundWork.detail ||
                backgroundWork.message ||
                backgroundWork.status ||
                backgroundWork.processingMessage ||
                backgroundWork.processing_message ||
                backgroundWork.agents?.[0]?.detail ||
                "Background agent is currently working"
            )
            : processingDetail;


    // ========================================================
    // VOICE PANEL VISIBILITY
    // ========================================================

    const voicePanelOpen =
        userSpeaking ||
        showingFinal ||
        sttDisplayActive;


    // ========================================================
    // VOICE STATUS
    // ========================================================

    let voiceStatus =
        "READY";


    if (
        userSpeaking
    ) {

        voiceStatus =
            "ACTIVE";

    } else if (
        showingFinal
    ) {

        voiceStatus =
            "FINAL";

    }


    // ========================================================
    // VOICE FOOTER
    // ========================================================

    const voiceFooter =
        showingFinal
            ? "FINAL INPUT"
            : userSpeaking
                ? "INPUT STREAM"
                : "READY";


    // ========================================================
    // STATE METADATA
    // ========================================================

    const stateMeta =
        STATE_META[
            zoeState
        ] ||
        STATE_META.idle;


    // ========================================================
    // RENDER ACTIVE PAGE
    // ========================================================

    function renderActivePage() {

        switch (
            activePage
        ) {

            case "chat":

                return (
                    <ChatPage
                        zoeState={
                            zoeState
                        }
                        online={
                            online
                        }
                        muted={
                            muted
                        }
                        response={
                            response
                        }
                        onToggleMute={
                            toggleMute
                        }
                        onClose={
                            closePage
                        }
                    />
                );


            default:

                return null;

        }

    }


    // ========================================================
    // RENDER
    // ========================================================

    return (
        <>
            <GlobalClickSound />

            <div
                className={
                    `zoe-dashboard ${activePage
                        ? "page-active"
                        : ""
                    }`
                }

                data-orb-state={
                    zoeState
                }

                data-active-page={
                    activePage ||
                    "dashboard"
                }

                data-processing-state={
                    effectiveProcessingState
                }

                data-background-processing={
                    backgroundProcessing
                        ? "true"
                        : "false"
                }
            >

                {/* ====================================================
                    BACKGROUND
                    ==================================================== */}

                <div className="zoe-bg-grid" />

                <div className="zoe-bg-vignette" />

                <div className="zoe-scanlines" />


                {/* ====================================================
                    CINEMATIC PROCESSING HUD
                    ==================================================== */}

                <ProcessingHUD
                    active={
                        processingActive
                    }

                    state={
                        effectiveProcessingState
                    }

                    operation={
                        effectiveProcessingOperation
                    }

                    detail={
                        effectiveProcessingDetail
                    }

                    background={
                        backgroundProcessing
                    }

                    agents={
                        backgroundWork?.agents || []
                    }
                />


                {/* ====================================================
                    DASHBOARD MODE
                    ==================================================== */}

                {
                    !activePage && (

                        <>

                            <ActivityStream
                                entries={
                                    activity
                                }
                            />


                            <NotificationStack
                                notifications={
                                    notifications
                                }
                            />


                            <SurfaceStage
                                surface={
                                    surface
                                }
                            />


                            {/* ====================================================
                                HEADER
                                ==================================================== */}

                            <header className="zoe-topbar">

                                <div className="zoe-brand">

                                    <span className="zoe-brand-mark">
                                        ◈
                                    </span>


                                    <div className="zoe-brand-text">

                                        <span className="zoe-brand-name">
                                            ZENITH ORCHESTRATION ENGINE
                                        </span>

                                    </div>

                                </div>


                                <div
                                    className={
                                        `zoe-status-pill ${online
                                            ? ""
                                            : "offline"
                                        }`
                                    }
                                >

                                    <span className="zoe-status-dot" />

                                    <span>

                                        {
                                            online
                                                ? stateMeta.label
                                                : "SYSTEM OFFLINE"
                                        }

                                    </span>

                                </div>


                                <Clock />

                            </header>


                            {/* ====================================================
                                MAIN DASHBOARD
                                ==================================================== */}

                            <main className="zoe-grid">

                                <section
                                    className="
                                        zoe-col
                                        zoe-col-center
                                    "
                                >

                                    <div className="zoe-orb-anchor" />


                                    {/* =================================================
                                        VOICE INPUT WINDOW
                                        ================================================= */}

                                    <div
                                        className={
                                            `zoe-voice-window ${voicePanelOpen
                                                ? "open"
                                                : ""
                                            }`
                                        }
                                    >

                                        <div className="zoe-voice-window-header">

                                            <div className="zoe-voice-window-title">

                                                <span className="zoe-voice-window-icon">
                                                    ◈
                                                </span>


                                                <div>

                                                    <div className="zoe-voice-window-name">
                                                        VOICE INPUT
                                                    </div>

                                                    <div className="zoe-voice-window-status">

                                                        {
                                                            showingFinal
                                                                ? "FINAL TRANSCRIPTION"
                                                                : userSpeaking
                                                                    ? "LIVE TRANSCRIPTION"
                                                                    : "VOICE INPUT"
                                                        }

                                                    </div>

                                                </div>

                                            </div>


                                            <div className="zoe-voice-window-indicator">

                                                <span />

                                                {
                                                    voiceStatus
                                                }

                                            </div>

                                        </div>


                                        <div className="zoe-voice-window-line" />


                                        <div className="zoe-voice-window-body">

                                            <div className="zoe-voice-wave">

                                                <span />
                                                <span />
                                                <span />
                                                <span />
                                                <span />
                                                <span />
                                                <span />

                                            </div>


                                            <div
                                                className={
                                                    `zoe-voice-text ${showingFinal
                                                        ? "final"
                                                        : ""
                                                    }`
                                                }
                                            >

                                                {
                                                    displayedStt
                                                }

                                            </div>

                                        </div>


                                        <div className="zoe-voice-window-footer">

                                            <span>
                                                {
                                                    voiceFooter
                                                }
                                            </span>

                                            <span>
                                                16 kHz / MONO
                                            </span>

                                        </div>


                                        <div className="zoe-voice-corner corner-tl" />
                                        <div className="zoe-voice-corner corner-tr" />
                                        <div className="zoe-voice-corner corner-bl" />
                                        <div className="zoe-voice-corner corner-br" />

                                    </div>


                                    {/* =================================================
                                        ZOE RESPONSE
                                        ================================================= */}

                                    {
                                        displayedResponse &&
                                        displayedResponse.trim() && (

                                            <button
                                                className="responsebox"
                                                type="button"
                                                onClick={() =>
                                                    openPage(
                                                        "chat"
                                                    )
                                                }
                                                aria-label="Open ZOE communications"
                                            >

                                                <span className="zoe-response">

                                                    {
                                                        displayedResponse
                                                    }


                                                    {/* =================================
                                                        GENERATION CURSOR
                                                        ================================= */}

                                                    <span
                                                        className="zoe-response-cursor"
                                                        aria-hidden="true"
                                                    />

                                                </span>


                                                {/* =====================================
                                                    RESPONSE COMMUNICATION INDICATOR
                                                    ===================================== */}

                                                <span
                                                    className="zoe-response-open-indicator"
                                                    aria-hidden="true"
                                                >

                                                    <span />
                                                    <span />
                                                    <span />

                                                </span>

                                            </button>

                                        )
                                    }


                                    {/* =================================================
                                        MICROPHONE
                                        ================================================= */}

                                    <button
                                        className={
                                            `zoe-mic ${muted
                                                ? "muted"
                                                : ""
                                            }`
                                        }

                                        onClick={
                                            toggleMute
                                        }

                                        aria-label={
                                            muted
                                                ? "Unmute ZOE"
                                                : "Mute ZOE"
                                        }

                                        type="button"
                                    >

                                        <span className="zoe-mic-icon">

                                            {
                                                muted
                                                    ? (
                                                        <MicrophoneOffIcon />
                                                    )
                                                    : (
                                                        <MicrophoneIcon />
                                                    )
                                            }

                                        </span>

                                    </button>

                                </section>

                            </main>

                        </>
                    )
                }


                {/* ====================================================
                    ACTIVE PAGE
                    ==================================================== */}

                {
                    activePage && (

                        <main className="zoe-page-shell">

                            <div className="zoe-page-content">

                                {
                                    renderActivePage()
                                }

                            </div>

                        </main>

                    )
                }


                {/* ====================================================
                    PERSISTENT ZOE ORB
                    ==================================================== */}

                <div
                    className={
                        `zoe-persistent-orb ${activePage
                            ? "page-mode"
                            : "dashboard-mode"
                        }`
                    }
                >

                    <ZoeOrb
                        label={
                            orbState
                        }

                        state={
                            orbState
                        }
                    />

                </div>

            </div>
        </>
    );

}


// ============================================================
// BACKGROUND WORK RESOLUTION
// ============================================================

function resolveBackgroundWork(
    data
) {

    const runtimeBgWork =
        data.runtime?.background_work;


    const topLevelBgWork =
        data.background_work ??
        data.backgroundWork;


    const explicit =
        runtimeBgWork ??
        topLevelBgWork ??
        null;


    if (
        explicit &&
        typeof explicit === "object" &&
        explicit.jobs &&
        Array.isArray(explicit.jobs)
    ) {

        const active =
            explicit.active === true;


        const agents =
            explicit.jobs.map(
                job => ({

                    id:
                        job.job_id,

                    name:
                        job.agent,

                    state:
                        job.status,

                    detail:
                        job.description ||
                        job.query ||
                        "",

                })
            );


        return {

            active,

            phase:
                explicit.phase ||
                "executing",

            count:
                explicit.count ||
                agents.length,

            jobs:
                explicit.jobs,

            agents,

            type:
                "agent",

            name:
                agents[0]?.name ||
                "",

            operation:
                agents[0]?.detail ||
                "",

            detail:
                agents[0]?.detail ||
                "",

            status:
                agents[0]?.state ||
                "",

            message:
                agents[0]?.detail ||
                "",

        };

    }


    if (
        explicit &&
        typeof explicit === "object"
    ) {

        const active =
            explicit.active === true;


        return {

            active,

            phase:
                explicit.phase ||
                "executing",

            count:
                1,

            jobs:
                [],

            agents:
                [],

            type:
                explicit.type ??
                "agent",

            name:
                explicit.name ??
                explicit.agent_name ??
                explicit.agentName ??
                "",

            operation:
                explicit.operation ??
                explicit.current_operation ??
                explicit.currentOperation ??
                "",

            detail:
                explicit.detail ??
                explicit.message ??
                explicit.status ??
                explicit.processing_message ??
                explicit.processingMessage ??
                "",

            status:
                explicit.status ??
                "",

            message:
                explicit.message ??
                "",

        };

    }


    const active =
        data.background_agent_active === true ||
        data.backgroundAgentActive === true ||
        data.agent_working === true ||
        data.agentWorking === true ||
        Number(
            data.active_agents ?? 0
        ) > 0 ||
        Number(
            data.activeAgents ?? 0
        ) > 0 ||
        (
            data.runtime &&
            Number(
                data.runtime.active_agents ?? 0
            ) > 0
        ) ||
        (
            data.runtime &&
            Number(
                data.runtime.active_agent_batches ?? 0
            ) > 0
        );


    if (
        !active
    ) {

        return null;

    }


    const runtimeJobs =
        data.runtime?.active_jobs;


    if (
        runtimeJobs &&
        Array.isArray(runtimeJobs) &&
        runtimeJobs.length > 0
    ) {

        const agents =
            runtimeJobs.map(
                job => ({

                    id:
                        job.job_id,

                    name:
                        job.agent,

                    state:
                        job.status,

                    detail:
                        job.description ||
                        job.query ||
                        "",

                })
            );


        return {

            active:
                true,

            phase:
                "executing",

            count:
                agents.length,

            jobs:
                runtimeJobs,

            agents,

            type:
                "agent",

            name:
                agents[0]?.name ||
                "",

            operation:
                agents[0]?.detail ||
                "",

            detail:
                agents[0]?.detail ||
                "",

            status:
                agents[0]?.state ||
                "",

            message:
                agents[0]?.detail ||
                "",

        };

    }


    const runtimeBatches =
        data.runtime?.agent_batches;


    if (
        runtimeBatches &&
        Array.isArray(runtimeBatches) &&
        runtimeBatches.length > 0
    ) {

        const agents = [];
        const jobs = [];


        for (
            const batch of runtimeBatches
        ) {

            if (
                batch.agents &&
                Array.isArray(
                    batch.agents
                )
            ) {

                for (
                    const a of batch.agents
                ) {

                    if (
                        a.status !==
                        "completed" &&
                        a.status !==
                        "failed"
                    ) {

                        agents.push({

                            id:
                                a.job_id,

                            name:
                                a.agent,

                            state:
                                a.status ||
                                "running",

                            detail:
                                a.query ||
                                "",

                        });


                        jobs.push({

                            job_id:
                                a.job_id,

                            batch_id:
                                batch.batch_id,

                            interaction_id:
                                batch.interaction_id,

                            agent:
                                a.agent,

                            query:
                                a.query,

                            description:
                                a.query,

                            status:
                                a.status ||
                                "running",

                            started_at:
                                batch.started_at,

                        });

                    }

                }

            }

        }


        if (
            agents.length > 0
        ) {

            return {

                active:
                    true,

                phase:
                    "executing",

                count:
                    agents.length,

                jobs,

                agents,

                type:
                    "agent",

                name:
                    agents[0]?.name ||
                    "",

                operation:
                    agents[0]?.detail ||
                    "",

                detail:
                    agents[0]?.detail ||
                    "",

                status:
                    agents[0]?.state ||
                    "",

                message:
                    agents[0]?.detail ||
                    "",

            };

        }

    }


    return {

        active:
            true,

        phase:
            "executing",

        count:
            1,

        jobs:
            [],

        agents:
            [],

        type:
            "agent",

        name:
            data.agent_name ??
            data.agentName ??
            "",

        operation:
            data.background_operation ??
            data.backgroundOperation ??
            data.current_operation ??
            data.currentOperation ??
            "",

        detail:
            data.background_detail ??
            data.backgroundDetail ??
            data.processing_message ??
            data.processingMessage ??
            data.status_message ??
            data.statusMessage ??
            "",

        status:
            data.status ??
            "",

        message:
            data.message ??
            "",

    };

}


// ============================================================
// BACKEND STATE RESOLUTION
// ============================================================

function getBackendState(
    data
) {

    if (
        data.error === true
    ) {

        return "error";

    }


    if (
        data.notifying === true
    ) {

        return "notifying";

    }


    if (
        data.executing === true
    ) {

        return "executing";

    }


    if (
        data.speaking === true
    ) {

        return "speaking";

    }


    if (
        data.thinking === true ||
        data.processing === true
    ) {

        return "thinking";

    }


    if (
        data.listening === true
    ) {

        return "listening";

    }


    if (
        data.muted === true
    ) {

        return "muted";

    }


    return "idle";

}


// ============================================================
// PROCESSING HUD STATE RESOLUTION
// ============================================================

function resolveProcessingMode(
    data
) {

    if (
        data.error === true
    ) {

        return "ERROR";

    }


    const explicit =
        String(
            data.processing_state ||
            data.processingState ||
            data.operation_type ||
            data.operationType ||
            ""
        ).toUpperCase();


    if (
        explicit.includes(
            "ERROR"
        ) ||
        explicit.includes(
            "FAULT"
        )
    ) {

        return "ERROR";

    }


    if (
        explicit.includes(
            "SUCCESS"
        ) ||
        explicit.includes(
            "COMPLETE"
        )
    ) {

        return "SUCCESS";

    }


    if (
        explicit.includes(
            "RESEARCH"
        )
    ) {

        return "RESEARCH";

    }


    if (
        explicit.includes(
            "AGENT"
        ) ||
        explicit.includes(
            "DELEGAT"
        )
    ) {

        return "AGENT";

    }


    if (
        explicit.includes(
            "TOOL"
        ) ||
        explicit.includes(
            "EXECUT"
        )
    ) {

        return "TOOL";

    }


    if (
        data.researching === true ||
        data.research === true
    ) {

        return "RESEARCH";

    }


    if (
        data.agent_working === true ||
        data.agentWorking === true ||
        data.delegating === true
    ) {

        return "AGENT";

    }


    if (
        data.executing === true ||
        data.tool_running === true ||
        data.toolRunning === true
    ) {

        return "TOOL";

    }


    if (
        data.success === true ||
        data.completed === true
    ) {

        return "SUCCESS";

    }


    return "PROCESSING";

}


// ============================================================
// CURRENT OPERATION
// ============================================================

function resolveProcessingOperation(
    data
) {

    const candidates = [

        data.current_operation,

        data.currentOperation,

        data.operation,

        data.processing_operation,

        data.processingOperation,

        data.tool_name,

        data.toolName,

        data.agent_name,

        data.agentName,

        data.current_task,

        data.currentTask,

        data.task_name,

        data.taskName,

    ];


    for (
        const candidate of candidates
    ) {

        if (
            typeof candidate === "string" &&
            candidate.trim()
        ) {

            return formatOperation(
                candidate
            );

        }

    }


    if (
        data.researching === true ||
        data.research === true
    ) {

        return "GATHERING INFORMATION";

    }


    if (
        data.agent_working === true ||
        data.agentWorking === true ||
        data.delegating === true
    ) {

        return "DELEGATING TO AGENT";

    }


    if (
        data.executing === true ||
        data.tool_running === true ||
        data.toolRunning === true
    ) {

        return "EXECUTING TOOL";

    }


    if (
        data.thinking === true
    ) {

        return "ANALYZING REQUEST";

    }


    if (
        data.processing === true
    ) {

        return "PROCESSING";

    }


    return "STANDING BY";

}


// ============================================================
// PROCESSING DETAIL
// ============================================================

function resolveProcessingDetail(
    data
) {

    const candidates = [

        data.processing_message,

        data.processingMessage,

        data.status_message,

        data.statusMessage,

        data.current_status,

        data.currentStatus,

        data.status,

        data.message,

    ];


    for (
        const candidate of candidates
    ) {

        if (
            typeof candidate === "string" &&
            candidate.trim()
        ) {

            return candidate.trim();

        }

    }


    return "";

}


// ============================================================
// OPERATION FORMATTING
// ============================================================

function formatOperation(
    value
) {

    return String(
        value
    )

        .replace(
            /[_-]+/g,
            " "
        )

        .replace(
            /\s+/g,
            " "
        )

        .trim()

        .toUpperCase();

}


// ============================================================
// CLOCK
// ============================================================

function Clock() {

    const [
        now,
        setNow,
    ] = useState(
        new Date()
    );


    useEffect(() => {

        const interval =
            setInterval(
                () => {

                    setNow(
                        new Date()
                    );

                },
                1000
            );


        return () => {

            clearInterval(
                interval
            );

        };

    }, []);


    const time =
        now.toLocaleTimeString(
            [],
            {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: false,
            }
        );


    const date =
        now.toLocaleDateString(
            [],
            {
                day: "2-digit",
                month: "short",
                year: "numeric",
            }
        ).toUpperCase();


    return (

        <div className="zoe-clock">

            <span className="zoe-time">

                {
                    time
                }

            </span>


            <span className="zoe-date">

                {
                    date
                }

            </span>

        </div>

    );

}
