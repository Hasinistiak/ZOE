import React, {
    useEffect,
    useRef,
    useState,
} from "react";
import "./ProcessingHUD.css";

/* ============================================================
   PROCESSING HUD
============================================================ */

export default function ProcessingHUD({
    active = false,
    state = "PROCESSING",
    operation = "",
    detail = "",
    background = false,
    agentName = "",
    agents = [],
}) {

    const startedAt = useRef(null);

    const [elapsed, setElapsed] = useState(0);


    /* ========================================================
       NORMALIZE AGENTS
    ======================================================== */

    const normalizedAgents = Array.isArray(agents)
        ? agents
            .filter(Boolean)
            .map((agent, index) => ({
                id:
                    agent.id ||
                    `${agent.name || "agent"}-${index}`,

                name:
                    agent.name ||
                    agent.agent ||
                    "agent",

                state:
                    agent.state ||
                    agent.status ||
                    "running",

                detail:
                    agent.detail ||
                    agent.operation ||
                    agent.message ||
                    "",
            }))
        : [];


    /* ========================================================
       ELAPSED TIMER
    ======================================================== */

    useEffect(() => {

        if (!active) {

            startedAt.current = null;

            setElapsed(0);

            return;
        }


        if (!startedAt.current) {

            startedAt.current =
                performance.now();
        }


        const timer =
            window.setInterval(() => {

                if (!startedAt.current) {
                    return;
                }

                setElapsed(
                    (performance.now() - startedAt.current) /
                    1000
                );

            }, 100);


        return () =>
            window.clearInterval(timer);

    }, [active]);


    /* ========================================================
       DON'T RENDER WHEN INACTIVE
    ======================================================== */

    if (!active) {
        return null;
    }


    /* ========================================================
       STATE
    ======================================================== */

    const mode =
        String(
            state || "PROCESSING"
        )
            .trim()
            .toUpperCase();


    /* ========================================================
       MODE LABEL
    ======================================================== */

    const modeLabel =
        background
            ? "BACKGROUND AGENT"

            : mode === "AGENT"
                ? "AGENT"

            : mode === "RESEARCH"
                ? "RESEARCH"

            : mode === "TOOL"
                ? "TOOL"

            : mode === "EXECUTING"
                ? "EXECUTION"

            : mode === "ERROR"
                ? "ERROR"

            : mode === "SUCCESS"
                ? "COMPLETE"

            : "COGNITIVE";


    /* ========================================================
       OPERATION
    ======================================================== */

    let operationLabel =
        String(operation || "").trim();


    if (!operationLabel) {

        operationLabel =
            background

                ? normalizedAgents.length > 0
                    ? normalizedAgents[0].name.toUpperCase()
                    : agentName
                        ? agentName.toUpperCase()
                        : "BACKGROUND WORK"

                : "ANALYZING REQUEST";
    }


    /* ========================================================
       DETAIL
    ======================================================== */

    let detailLabel =
        String(detail || "").trim();


    if (!detailLabel) {

        detailLabel =
            background

                ? normalizedAgents.length > 0

                    ? `${normalizedAgents.length} agent${
                        normalizedAgents.length === 1
                            ? ""
                            : "s"
                    } currently working`

                    : "Background agent is currently working"

                : "ZOE cognitive runtime active";
    }


    /* ========================================================
       CSS STATE
    ======================================================== */

    const cssState =

        mode === "ERROR"
            ? "error"

            : mode === "SUCCESS"
                ? "success"

                : background
                    ? "agent"

                    : mode.toLowerCase();


    /* ========================================================
       AGENT COUNT
    ======================================================== */

    const agentCount =
        normalizedAgents.length;


    /* ========================================================
       RENDER
    ======================================================== */

    return (
        <div
            className={`processing-hud processing-hud--${cssState}`}
            aria-hidden="true"
        >

            <div className="processing-hud__rail">


                {/* ==================================================
                    HEADER
                ================================================== */}

                <div className="processing-hud__header">

                    <div className="processing-hud__identity">

                        <span className="processing-hud__indicator" />

                        <span className="processing-hud__system">
                            ZOE
                        </span>

                        <span className="processing-hud__separator">
                            /
                        </span>

                        <span className="processing-hud__module">
                            {background
                                ? "BACKGROUND RUNTIME"
                                : "COGNITIVE CORE"}
                        </span>

                    </div>


                    <div className="processing-hud__state">

                        <span>
                            {modeLabel}
                        </span>

                        <i />

                    </div>

                </div>


                {/* ==================================================
                    MAIN OPERATION
                ================================================== */}

                <div className="processing-hud__main">

                    <div className="processing-hud__operation">

                        <span className="processing-hud__label">
                            CURRENT OPERATION
                        </span>

                        <div className="processing-hud__operation-value">
                            {operationLabel}
                        </div>

                        <div className="processing-hud__detail">
                            {detailLabel}
                        </div>

                    </div>


                    {/* ==================================================
                        ELAPSED
                    ================================================== */}

                    <div className="processing-hud__elapsed">

                        <span className="processing-hud__label">
                            ACTIVE
                        </span>

                        <strong>
                            {elapsed.toFixed(1)}
                        </strong>

                        <small>
                            SEC
                        </small>

                    </div>

                </div>


                {/* ==================================================
                    ACTIVE AGENTS
                ================================================== */}

                {background && agentCount > 0 && (

                    <div className="processing-hud__agents">


                        <div className="processing-hud__agents-header">

                            <span>
                                ACTIVE AGENTS
                            </span>

                            <strong>
                                {String(agentCount).padStart(2, "0")}
                            </strong>

                        </div>


                        <div className="processing-hud__agent-list">

                            {normalizedAgents.map(
                                (agent) => (

                                    <div
                                        className="processing-hud__agent"
                                        key={agent.id}
                                    >

                                        <div className="processing-hud__agent-top">


                                            <div className="processing-hud__agent-name">

                                                <span className="processing-hud__agent-dot" />

                                                <span>
                                                    {String(
                                                        agent.name
                                                    ).toUpperCase()}
                                                </span>

                                            </div>


                                            <span className="processing-hud__agent-state">

                                                {String(
                                                    agent.state
                                                ).toUpperCase()}

                                            </span>

                                        </div>


                                        {agent.detail && (

                                            <div className="processing-hud__agent-detail">
                                                {agent.detail}
                                            </div>

                                        )}

                                    </div>

                                )
                            )}

                        </div>

                    </div>

                )}


                {/* ==================================================
                    NO AGENTS YET
                ================================================== */}

                {background && agentCount === 0 && (

                    <div className="processing-hud__agent-empty">

                        <span className="processing-hud__agent-empty-dot" />

                        <span>
                            INITIALIZING BACKGROUND WORK
                        </span>

                    </div>

                )}


                {/* ==================================================
                    PIPELINE
                ================================================== */}

                <div className="processing-hud__pipeline">

                    <span className="processing-hud__pipeline-item">
                        INPUT
                    </span>

                    <i />

                    <span className="processing-hud__pipeline-item active">
                        ROUTE
                    </span>

                    <i />

                    <span className="processing-hud__pipeline-item active">
                        {background
                            ? "AGENT"
                            : "CORE"}
                    </span>

                    <i />

                    <span className="processing-hud__pipeline-item active">
                        EXECUTE
                    </span>

                    <i />

                    <span className="processing-hud__pipeline-item">
                        OUTPUT
                    </span>

                </div>

            </div>

        </div>
    );
}