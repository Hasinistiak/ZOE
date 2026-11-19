const clickSound = new Audio("/sounds/click.mp3");

clickSound.volume = 0.2;

export function playClick() {
    clickSound.currentTime = 0;
    clickSound.play().catch(() => {});
}