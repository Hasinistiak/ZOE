
import React, {
    useCallback,
    useEffect,
    useRef,
    useState,
} from "react";

import "./ChatPage.css";

export default function ChatPage({
    zoeState = "idle",
    online = false,
    response = "",
    onClose,
}) {


    const API_BASE = import.meta.env.DEV
        ? ""
        : "http://127.0.0.1:8000";

    // ========================================================
    // LOCAL CHAT STATE
    // ========================================================

    const [
        input,
        setInput,
    ] = useState("");

    const [
        sending,
        setSending,
    ] = useState(false);

    const [
        messages,
        setMessages,
    ] = useState([]);


    // ========================================================
    // REFS
    // ========================================================

    const messagesEndRef =
        useRef(null);

    const mountedRef =
        useRef(true);

    const requestIdRef =
        useRef(0);


    // ========================================================
    // RESPONSE SYNC REFS
    // ========================================================

    const lastDashboardResponseRef =
        useRef("");

    /*
     * Prevent the same response from being inserted twice
     * through different channels.
     *
     * We deliberately use a timestamped record instead of a
     * permanent Set because the same text may legitimately
     * occur later in a different conversation turn.
     */
    const recentAssistantResponseRef =
        useRef({
            text: "",
            timestamp: 0,
        });


    // ========================================================
    // CLEANUP
    // ========================================================

    useEffect(() => {

        mountedRef.current =
            true;

        return () => {

            mountedRef.current =
                false;

        };

    }, []);


    // ========================================================
    // ASSISTANT MESSAGE APPENDER
    // ========================================================

    /*
     * All assistant responses go through this function.
     *
     * This is important:
     *
     *   API response
     *        \
     *         -> appendAssistantMessage()
     *        /
     *   dashboard response
     *
     * That gives us one place to prevent duplicates.
     */

    const appendAssistantMessage =
        useCallback((text) => {

            const normalized =
                typeof text === "string"
                    ? text.trim()
                    : "";

            if (!normalized) {
                return false;
            }


            if (!mountedRef.current) {
                return false;
            }


            const now =
                Date.now();

            const recent =
                recentAssistantResponseRef.current;


            /*
             * If exactly the same assistant response was
             * already inserted very recently, ignore it.
             *
             * 1500ms is long enough to catch the API-response
             * + dashboard-sync race without preventing the
             * same sentence from appearing naturally later.
             */
            if (
                recent.text === normalized &&
                now - recent.timestamp < 1500
            ) {

                return false;

            }


            recentAssistantResponseRef.current = {
                text: normalized,
                timestamp: now,
            };


            setMessages(prev => {

                /*
                 * Extra protection:
                 *
                 * If React state already contains the exact
                 * same assistant message as the newest message,
                 * don't append another one.
                 */
                const last =
                    prev[prev.length - 1];

                if (
                    last &&
                    last.role === "assistant" &&
                    typeof last.text === "string" &&
                    last.text.trim() === normalized
                ) {

                    return prev;

                }


                return [

                    ...prev,

                    {
                        id:
                            `${now}-assistant-${Math.random()
                                .toString(36)
                                .slice(2, 9)}`,

                        role:
                            "assistant",

                        text:
                            normalized,
                    },

                ];

            });


            return true;

        }, []);


    // ========================================================
    // DASHBOARD RESPONSE SYNC
    // ========================================================

    useEffect(() => {

        const text =
            typeof response === "string"
                ? response.trim()
                : "";

        if (!text) {
            return;
        }


        /*
         * React can re-run effects with the same prop value.
         *
         * Do not process the same dashboard response twice.
         */
        if (
            text ===
            lastDashboardResponseRef.current
        ) {

            return;

        }


        lastDashboardResponseRef.current =
            text;


        appendAssistantMessage(
            text
        );

    }, [
        response,
        appendAssistantMessage,
    ]);


    // ========================================================
    // AUTO SCROLL
    // ========================================================

    useEffect(() => {

        messagesEndRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "end",
        });

    }, [
        messages,
        sending,
    ]);


    // ========================================================
    // RESPONSE EXTRACTION
    // ========================================================

    function getResponseText(result) {

        if (
            result == null
        ) {

            return "";

        }


        // ====================================================
        // NORMAL ZOE RESPONSE
        // ====================================================

        if (
            typeof result === "object" &&
            result.type === "response"
        ) {

            return (
                result.text ||
                ""
            );

        }


        // ====================================================
        // BACKGROUND TASK
        // ====================================================

        if (
            typeof result === "object" &&
            result.type === "background_task"
        ) {

            return (
                result.text ||
                "I'm on it."
            );

        }


        // ====================================================
        // TASK
        // ====================================================

        if (
            typeof result === "object" &&
            result.type === "task"
        ) {

            return (
                result.acknowledgement ||
                result.text ||
                "I'm on it."
            );

        }


        // ====================================================
        // STATUS
        // ====================================================

        if (
            typeof result === "object" &&
            result.type === "status"
        ) {

            return (
                result.text ||
                result.message ||
                ""
            );

        }


        // ====================================================
        // ERROR
        // ====================================================

        if (
            typeof result === "object" &&
            result.type === "error"
        ) {

            return (
                result.text ||
                "Something went wrong."
            );

        }


        // ====================================================
        // STRING
        // ====================================================

        if (
            typeof result === "string"
        ) {

            return result;

        }


        return "";

    }


    // ========================================================
    // SEND MESSAGE
    // ========================================================

    async function sendMessage() {

        const text =
            input.trim();


        if (
            !text ||
            sending
        ) {

            return;

        }


        const requestId =
            ++requestIdRef.current;


        // ====================================================
        // CLEAR INPUT
        // ====================================================

        setInput("");

        setSending(true);


        // ====================================================
        // ADD USER MESSAGE
        // ====================================================

        setMessages(prev => [

            ...prev,

            {
                id:
                    `${Date.now()}-user-${requestId}`,

                role:
                    "user",

                text,
            },

        ]);


        try {

            const res =
                await fetch(
                    `${API_BASE}/api/chat`,
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json",
                        },

                        body:
                            JSON.stringify({
                                message:
                                    text,
                            }),
                    }
                );


            let data;

            try {

                data =
                    await res.json();

            } catch {

                data =
                    null;

            }


            if (
                !res.ok ||
                !data?.success
            ) {

                throw new Error(
                    data?.error ||
                    `HTTP ${res.status}`
                );

            }


            // =================================================
            // BACKEND RETURNS:
            //
            // {
            //     success: true,
            //     result: {...}
            // }
            // =================================================

            const responseText =
                getResponseText(
                    data.result
                ).trim();


            if (
                !mountedRef.current ||
                requestId !==
                requestIdRef.current
            ) {

                return;

            }


            if (responseText) {

                /*
                 * IMPORTANT:
                 *
                 * Do NOT directly call setMessages here.
                 *
                 * Route the response through the same function
                 * used by the dashboard response path.
                 */
                appendAssistantMessage(
                    responseText
                );

            }

        } catch (error) {

            console.error(
                "[ZOE] Chat failed:",
                error
            );


            if (
                !mountedRef.current ||
                requestId !==
                requestIdRef.current
            ) {

                return;

            }


            setMessages(prev => [

                ...prev,

                {
                    id:
                        `${Date.now()}-error-${requestId}`,

                    role:
                        "assistant",

                    text:
                        "Unable to reach ZOE.",
                },

            ]);

        } finally {

            if (
                mountedRef.current &&
                requestId ===
                requestIdRef.current
            ) {

                setSending(false);

            }

        }

    }


    // ========================================================
    // KEYBOARD
    // ========================================================

    function handleKeyDown(
        event
    ) {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();

        }

    }


    // ========================================================
    // CLOSE
    // ========================================================

    function handleClose() {

        if (
            typeof onClose ===
            "function"
        ) {

            onClose();

        }

    }


    // ========================================================
    // RENDER
    // ========================================================

    return (

        <div
            className="zoe-chat-page"
            data-orb-state={
                zoeState
            }
        >

            {/* =================================================
                BACKGROUND
            ================================================= */}

            <div className="chat-grid" />

            <div className="chat-vignette" />

            <div className="chat-scanlines" />


            {/* =================================================
                HEADER
            ================================================= */}

            <header className="chat-header">

                <div className="chat-brand">

                    <span className="zoe-brand-mark">
                        ◈
                    </span>


                    <div>

                        <span className="zoe-brand-name">
                            ZENITH ORCHESTRATION ENGINE
                        </span>




                    </div>

                </div>


                <div className="chat-header-actions">



                    <button
                        type="button"
                        className="chat-back"
                        onClick={
                            handleClose
                        }
                        aria-label="Return to dashboard"
                    >
                        ← DASHBOARD
                    </button>

                </div>

            </header>


            {/* =================================================
                CONTENT
            ================================================= */}

            <main className="chat-content">

                <section className="chat-messages">

                    {
                        messages.map(
                            message => (

                                <div
                                    key={
                                        message.id
                                    }
                                    className={
                                        `chat-message ${message.role ===
                                            "user"
                                            ? "user"
                                            : "assistant"
                                        }`
                                    }
                                >

                                    <div className="message-label">

                                        {
                                            message.role ===
                                                "user"
                                                ? "YOU"
                                                : "ZOE"
                                        }

                                    </div>


                                    <div className="message-text">

                                        {
                                            message.text
                                        }

                                    </div>

                                </div>

                            )
                        )
                    }


                    {/* =================================================
                        THINKING
                    ================================================= */}

                    {
                        sending && (

                            <div className="chat-message assistant">

                                <div className="message-label">
                                    ZOE
                                </div>


                                <div className="message-text thinking">

                                    <span />
                                    <span />
                                    <span />

                                </div>

                            </div>

                        )
                    }


                    <div
                        ref={
                            messagesEndRef
                        }
                    />

                </section>

            </main>


            {/* =================================================
                COMPOSER
            ================================================= */}

            <footer className="chat-composer">

                <div className="chat-input-shell">

                    {/* =================================================
                        INPUT
                    ================================================= */}

                    <input
                        value={
                            input
                        }
                        onChange={
                            event =>
                                setInput(
                                    event.target.value
                                )
                        }
                        onKeyDown={
                            handleKeyDown
                        }
                        placeholder="Message ZOE..."
                        autoComplete="off"
                        disabled={
                            sending
                        }
                        aria-label="Message ZOE"
                    />


                    {/* =================================================
                        SEND
                    ================================================= */}

                    <button
                        type="button"
                        className="chat-send"
                        onClick={
                            sendMessage
                        }
                        disabled={
                            !input.trim() ||
                            sending
                        }
                        aria-label="Send message"
                    >

                        {
                            sending
                                ? "..."
                                : "↑"
                        }

                    </button>

                </div>


                {/* =================================================
                    FOOTER
                ================================================= */}

                <div className="chat-footer">

                    <span className="chat-footer-dot" />

                    ZOE

                    <span>•</span>

                    MARK 33

                </div>

            </footer>

        </div>

    );

}
