/**
 * BioSync Attendance - Main Module
 */

document.addEventListener('DOMContentLoaded', () => {
    initToasts();
    initAnimations();
});

/**
 * Handle auto-dismissal of flash messages
 */
function initToasts() {
    const flashes = document.querySelectorAll('.flash-container .glass-panel');
    flashes.forEach(flash => {
        setTimeout(() => {
            flash.style.opacity = '0';
            flash.style.transform = 'translateY(-10px)';
            flash.style.transition = 'opacity 0.5s, transform 0.5s';
            setTimeout(() => {
                flash.remove();
            }, 500);
        }, 5000);
    });
}

/**
 * Initialize staggered animations for list items
 */
function initAnimations() {
    const listItems = document.querySelectorAll('.animate-list > *');
    listItems.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(10px)';
        item.style.transition = `opacity 0.4s ease ${index * 0.1}s, transform 0.4s ease ${index * 0.1}s`;

        requestAnimationFrame(() => {
            item.style.opacity = '1';
            item.style.transform = 'translateY(0)';
        });
    });
}
