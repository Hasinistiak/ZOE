
// ZoeSystems.jsx

import React, {
    useCallback,
    useEffect,
    useRef,
    useState,
} from "react";

import "./ZoeSystems.css";


// ============================================================
// SURFACE ICONS
// ============================================================

const SURFACE_ICONS = {
    calendar: "◷",
    spotify: "♪",
    weather: "◐",
    reminder: "◈",
    generic: "◇",
};


// ============================================================
// SURFACE STAGE
// ============================================================

export function SurfaceStage({
    surface,
}) {

    const [
        mounted,
        setMounted,
    ] = useState(null);

    const [
        closing,
        setClosing,
    ] = useState(false);

    const dismissTimer =
        useRef(null);

    const closeTimer =
        useRef(null);

    const mountedRef =
        useRef(null);


    const clearTimers =
        useCallback(() => {

            if (
                dismissTimer.current
            ) {

                clearTimeout(
                    dismissTimer.current
                );

                dismissTimer.current =
                    null;

            }


            if (
                closeTimer.current
            ) {

                clearTimeout(
                    closeTimer.current
                );

                closeTimer.current =
                    null;

            }

        }, []);


    const closeSurface =
        useCallback(() => {

            clearTimers();

            setClosing(true);


            closeTimer.current =
                setTimeout(() => {

                    mountedRef.current =
                        null;

                    setMounted(
                        null
                    );

                    setClosing(
                        false
                    );

                    closeTimer.current =
                        null;

                }, 320);

        }, [
            clearTimers,
        ]);


    useEffect(() => {

        clearTimers();


        if (
            !surface
        ) {

            if (
                mountedRef.current
            ) {

                closeSurface();

            }

            return;

        }


        mountedRef.current =
            surface;

        setMounted(
            surface
        );

        setClosing(
            false
        );


        if (
            !surface.pinned
        ) {

            const ttl =
                typeof surface.ttl === "number"
                    ? Math.max(
                        500,
                        surface.ttl
                    )
                    : 12000;


            dismissTimer.current =
                setTimeout(() => {

                    setClosing(
                        true
                    );


                    closeTimer.current =
                        setTimeout(() => {

                            mountedRef.current =
                                null;

                            setMounted(
                                null
                            );

                            setClosing(
                                false
                            );

                            closeTimer.current =
                                null;

                        }, 320);


                    dismissTimer.current =
                        null;

                }, ttl);

        }


        return clearTimers;

    }, [
        surface,
        clearTimers,
        closeSurface,
    ]);


    useEffect(() => {

        return () =>
            clearTimers();

    }, [
        clearTimers,
    ]);


    if (
        !mounted
    ) {

        return null;

    }


    const icon =
        SURFACE_ICONS[
            mounted.type
        ] ||
        SURFACE_ICONS.generic;


    const rows =
        Array.isArray(
            mounted.rows
        )
            ? mounted.rows
            : [];


    return (

        <div
            className={
                `zoe-surface ${
                    closing
                        ? "closing"
                        : "opening"
                }`
            }
        >

            <div className="zoe-surface-header">

                <span className="zoe-surface-icon">
                    {icon}
                </span>


                <div className="zoe-surface-title">

                    <span className="zoe-surface-type">
                        {
                            mounted.type ||
                            "SYSTEM"
                        }
                    </span>

                    <span className="zoe-surface-name">
                        {
                            mounted.title ||
                            "Information"
                        }
                    </span>

                </div>

            </div>


            <div className="zoe-surface-body">

                {
                    rows.map(
                        (
                            row,
                            index
                        ) => (

                            <div
                                className="zoe-surface-row"
                                key={
                                    row.id ||
                                    `${row.label || "row"}_${index}`
                                }
                            >

                                <span className="zoe-surface-row-label">
                                    {row.label}
                                </span>

                                <span className="zoe-surface-row-value">
                                    {row.value}
                                </span>

                            </div>

                        )
                    )
                }


                {
                    rows.length === 0 && (

                        <div className="zoe-surface-empty">
                            No data attached to this surface.
                        </div>

                    )
                }

            </div>


            <div className="zoe-surface-edge" />

        </div>

    );

}


// ============================================================
// NOTIFICATIONS
// ============================================================

const TIER_DURATION = {
    critical: null,
    important: 8000,
    informational: 5000,
    ambient: 3000,
};


const MAX_NOTIFICATIONS = 6;


export function NotificationStack({
    notifications = [],
}) {

    const [
        items,
        setItems,
    ] = useState([]);


    const seen =
        useRef(
            new Set()
        );


    const timers =
        useRef(
            new Map()
        );


    const clearTimer =
        useCallback(
            (id) => {

                const timer =
                    timers.current.get(
                        id
                    );


                if (timer) {

                    clearTimeout(
                        timer
                    );

                    timers.current.delete(
                        id
                    );

                }

            },
            []
        );


    const dismiss =
        useCallback(
            (id) => {

                clearTimer(
                    id
                );


                setItems(
                    previous =>
                        previous.filter(
                            item =>
                                item.id !== id
                        )
                );

            },
            [
                clearTimer,
            ]
        );


    useEffect(() => {

        if (
            !Array.isArray(
                notifications
            )
        ) {

            return;

        }


        if (
            notifications.length === 0
        ) {

            return;

        }


        const fresh =
            notifications.filter(
                notification =>
                    notification &&
                    notification.id &&
                    !seen.current.has(
                        notification.id
                    )
            );


        if (
            fresh.length === 0
        ) {

            return;

        }


        fresh.forEach(
            notification => {

                seen.current.add(
                    notification.id
                );


                const duration =
                    TIER_DURATION[
                        notification.tier
                    ] ??
                    TIER_DURATION.informational;


                if (
                    duration !== null
                ) {

                    const timer =
                        setTimeout(
                            () => {

                                setItems(
                                    previous =>
                                        previous.filter(
                                            item =>
                                                item.id !==
                                                notification.id
                                        )
                                );


                                timers.current.delete(
                                    notification.id
                                );

                            },
                            duration
                        );


                    timers.current.set(
                        notification.id,
                        timer
                    );

                }

            }
        );


        setItems(
            previous => {

                const merged = [
                    ...previous,
                    ...fresh,
                ];


                return merged.slice(
                    -MAX_NOTIFICATIONS
                );

            }
        );

    }, [
        notifications,
    ]);


    useEffect(() => {

        return () => {

            for (
                const timer
                of timers.current.values()
            ) {

                clearTimeout(
                    timer
                );

            }


            timers.current.clear();

        };

    }, []);


    if (
        items.length === 0
    ) {

        return null;

    }


    return (

        <div className="zoe-notify-stack">

            {
                items.map(
                    notification => {

                        const tier =
                            notification.tier ||
                            "informational";


                        return (

                            <button
                                className={
                                    `zoe-notify zoe-notify-${tier}`
                                }
                                key={
                                    notification.id
                                }
                                onClick={() =>
                                    dismiss(
                                        notification.id
                                    )
                                }
                                type="button"
                            >

                                <span className="zoe-notify-tier">
                                    {tier}
                                </span>


                                <span className="zoe-notify-message">
                                    {
                                        notification.message ||
                                        notification.text ||
                                        ""
                                    }
                                </span>


                                <span className="zoe-notify-dismiss">
                                    ×
                                </span>

                            </button>

                        );

                    }
                )
            }

        </div>

    );

}


// ============================================================
// ACTIVITY STREAM
// COMPACT AUTO-SCROLLING TELEMETRY WIDGET
// ============================================================

export function ActivityStream({
    entries = [],
}) {

    const safeEntries =
        Array.isArray(entries)
            ? entries.filter(Boolean)
            : [];


    const scrollRef =
        useRef(null);


    const previousLength =
        useRef(
            safeEntries.length
        );


    const [
        activeIndex,
        setActiveIndex,
    ] = useState(
        Math.max(
            0,
            safeEntries.length - 1
        )
    );


    // ========================================================
    // TIME FORMATTER
    // ========================================================

    const formatTime = (
        entry
    ) => {

        if (
            entry?.time &&
            typeof entry.time === "string" &&
            !entry.time.includes("T")
        ) {

            return entry.time;

        }


        const timestamp =
            entry?.timestamp ||
            entry?.time;


        if (!timestamp) {

            return "";

        }


        try {

            const date =
                new Date(
                    timestamp
                );


            if (
                Number.isNaN(
                    date.getTime()
                )
            ) {

                return "";

            }


            return date.toLocaleTimeString(
                [],
                {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                }
            );

        } catch {

            return "";

        }

    };


    // ========================================================
    // ICON
    // ========================================================

    const getIcon = (
        entry
    ) => {

        return (
            SURFACE_ICONS[
                entry?.surface
            ] ||
            SURFACE_ICONS.generic
        );

    };


    // ========================================================
    // TYPE
    // ========================================================

    const getType = (
        entry
    ) => {

        if (
            entry?.kind === "agent"
        ) {

            return "AGENT";

        }


        if (
            entry?.kind === "error"
        ) {

            return "ERROR";

        }


        if (
            entry?.kind === "warning"
        ) {

            return "WARN";

        }


        if (
            entry?.kind === "success"
        ) {

            return "OK";

        }


        return (
            entry?.surface ||
            "SYSTEM"
        ).toUpperCase();

    };


    // ========================================================
    // NEW EVENTS
    // ONLY SWITCH TO THE NEWEST EVENT
    // ========================================================

    useEffect(() => {

        if (
            safeEntries.length === 0
        ) {

            setActiveIndex(
                0
            );

            previousLength.current =
                0;

            return;

        }


        if (
            safeEntries.length >
            previousLength.current
        ) {

            setActiveIndex(
                safeEntries.length - 1
            );

        }


        previousLength.current =
            safeEntries.length;

    }, [
        safeEntries.length,
    ]);


    // ========================================================
    // SCROLL ACTIVE EVENT INTO VIEW
    // ========================================================

    useEffect(() => {

        if (
            !scrollRef.current
        ) {

            return;

        }


        const container =
            scrollRef.current;


        const item =
            container.children[
                activeIndex
            ];


        if (
            item
        ) {

            item.scrollIntoView({
                behavior: "smooth",
                block: "nearest",
            });

        }

    }, [
        activeIndex,
    ]);


    // ========================================================
    // EMPTY STATE
    // ========================================================

    if (
        safeEntries.length === 0
    ) {

        return (

            <div className="zoe-activity-widget zoe-activity-idle">

                <div className="zoe-activity-widget-top">

                    <div className="zoe-activity-widget-heading">

                        <span className="zoe-activity-live-dot" />

                        <span className="zoe-activity-widget-title">
                            ZOE ACTIVITY
                        </span>

                    </div>


                    <span className="zoe-activity-widget-status">
                        IDLE
                    </span>

                </div>


                <div className="zoe-activity-idle-content">

                    <span className="zoe-activity-idle-icon">
                        ◉
                    </span>


                    <span>
                        SYSTEM READY
                    </span>

                </div>


                <div className="zoe-activity-widget-footer">

                    <span className="zoe-activity-footer-line" />

                    <span>
                        LIVE TELEMETRY
                    </span>

                    <span className="zoe-activity-footer-pulse">
                        ●
                    </span>

                </div>

            </div>

        );

    }


    // ========================================================
    // RENDER
    // ========================================================

    return (

        <div className="zoe-activity-widget">

            <div className="zoe-activity-widget-top">

                <div className="zoe-activity-widget-heading">

                    <span className="zoe-activity-live-dot" />

                    <span className="zoe-activity-widget-title">
                        ZOE ACTIVITY
                    </span>

                </div>


                <div className="zoe-activity-widget-meta">

                    {
                        safeEntries.length
                    }

                    {" "}

                    EVENTS

                </div>

            </div>


            <div
                className="zoe-activity-widget-scroll"
                ref={scrollRef}
            >

                {
                    safeEntries.map(
                        (
                            entry,
                            index
                        ) => {

                            const icon =
                                getIcon(
                                    entry
                                );


                            const type =
                                getType(
                                    entry
                                );


                            const time =
                                formatTime(
                                    entry
                                );


                            const label =
                                entry.label ||
                                "SYSTEM";


                            const text =
                                entry.text ||
                                entry.message ||
                                "";


                            const isActive =
                                index ===
                                activeIndex;


                            return (

                                <div
                                    className={
                                        `zoe-activity-widget-entry ${
                                            isActive
                                                ? "active"
                                                : ""
                                        } ${
                                            entry.kind
                                                ? `kind-${entry.kind}`
                                                : ""
                                        } ${
                                            entry.status
                                                ? `status-${entry.status}`
                                                : ""
                                        }`
                                    }
                                    key={
                                        entry.id ||
                                        `activity_${index}`
                                    }
                                >

                                    <div className="zoe-activity-widget-icon">
                                        {icon}
                                    </div>


                                    <div className="zoe-activity-widget-content">

                                        <div className="zoe-activity-widget-line">

                                            <span className="zoe-activity-widget-type">
                                                {type}
                                            </span>


                                            <span className="zoe-activity-widget-label">
                                                {label}
                                            </span>


                                            {
                                                time && (

                                                    <span className="zoe-activity-widget-time">
                                                        {time}
                                                    </span>

                                                )
                                            }

                                        </div>


                                        <div className="zoe-activity-widget-text">

                                            {text}

                                        </div>


                                        {
                                            entry.status &&
                                            entry.kind === "agent" && (

                                                <div className="zoe-activity-widget-agent-status">

                                                    {
                                                        String(
                                                            entry.status
                                                        ).toUpperCase()
                                                    }

                                                </div>

                                            )
                                        }

                                    </div>


                                    <div className="zoe-activity-widget-marker" />

                                </div>

                            );

                        }
                    )
                }

            </div>


            <div className="zoe-activity-widget-footer">

                <span className="zoe-activity-footer-line" />

                <span>
                    LIVE TELEMETRY
                </span>

                <span className="zoe-activity-footer-pulse">
                    ●
                </span>

            </div>

        </div>

    );

}
