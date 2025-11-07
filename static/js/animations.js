// Advanced Loading States and Success Animations for Cardia4

class AnimationManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupScrollReveal();
        this.setupFormAnimations();
        this.setupNotifications();
        this.setupFileUpload();
    }

    // Loading States
    showLoading(element, text = 'Processing...') {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        element.classList.add('btn-loading');
        element.disabled = true;
        element.setAttribute('data-original-text', element.textContent);
        element.textContent = text;
    }

    hideLoading(element) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }
        
        element.classList.remove('btn-loading');
        element.disabled = false;
        const originalText = element.getAttribute('data-original-text');
        if (originalText) {
            element.textContent = originalText;
            element.removeAttribute('data-original-text');
        }
    }

    // Success Animations
    showSuccess(element, message = 'Success!', duration = 3000) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }

        element.classList.add('success-animation');
        
        // Create success notification
        this.showNotification(message, 'success', duration);
        
        // Add checkmark animation
        const checkmark = document.createElement('i');
        checkmark.setAttribute('data-lucide', 'check-circle');
        checkmark.className = 'w-5 h-5 success-checkmark';
        
        const originalContent = element.innerHTML;
        element.innerHTML = '';
        element.appendChild(checkmark);
        element.appendChild(document.createTextNode(' ' + message));
        
        // Initialize lucide icons
        if (window.lucide) {
            lucide.createIcons();
        }
        
        setTimeout(() => {
            element.classList.remove('success-animation');
            element.innerHTML = originalContent;
            if (window.lucide) {
                lucide.createIcons();
            }
        }, duration);
    }

    // Progress Bar Animation
    animateProgress(element, targetValue, duration = 1000) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }

        let startValue = 0;
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const currentValue = startValue + (targetValue - startValue) * this.easeOutCubic(progress);
            
            element.style.width = currentValue + '%';
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    // Easing function
    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    // Notification System
    showNotification(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="flex items-center gap-3">
                <i data-lucide="${this.getNotificationIcon(type)}" class="w-5 h-5"></i>
                <span>${message}</span>
                <button class="ml-4 opacity-70 hover:opacity-100" onclick="this.parentElement.parentElement.remove()">
                    <i data-lucide="x" class="w-4 h-4"></i>
                </button>
            </div>
        `;

        document.body.appendChild(notification);
        
        if (window.lucide) {
            lucide.createIcons();
        }

        setTimeout(() => {
            if (notification.parentElement) {
                notification.style.animation = 'fadeOut 0.3s ease-out';
                setTimeout(() => notification.remove(), 300);
            }
        }, duration);
    }

    getNotificationIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'alert-circle',
            warning: 'alert-triangle',
            info: 'info'
        };
        return icons[type] || 'info';
    }

    // Scroll Reveal Animation
    setupScrollReveal() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                }
            });
        }, observerOptions);

        // Observe elements with scroll-reveal class
        document.querySelectorAll('.scroll-reveal').forEach(el => {
            observer.observe(el);
        });
    }

    // Form Animation Enhancements
    setupFormAnimations() {
        // Add floating label effect
        document.querySelectorAll('.form-input').forEach(input => {
            input.addEventListener('focus', () => {
                input.parentElement.classList.add('focused');
            });

            input.addEventListener('blur', () => {
                if (!input.value) {
                    input.parentElement.classList.remove('focused');
                }
            });
        });

        // Form submission animations
        document.querySelectorAll('form').forEach(form => {
            form.addEventListener('submit', (e) => {
                const submitBtn = form.querySelector('button[type="submit"]');
                if (submitBtn && !submitBtn.classList.contains('btn-loading')) {
                    this.showLoading(submitBtn);
                }
            });
        });
    }

    // File Upload Animations
    setupFileUpload() {
        document.querySelectorAll('input[type="file"]').forEach(input => {
            const dropArea = input.closest('.file-upload-area') || input.parentElement;
            
            if (dropArea) {
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    dropArea.addEventListener(eventName, this.preventDefaults, false);
                });

                ['dragenter', 'dragover'].forEach(eventName => {
                    dropArea.addEventListener(eventName, () => dropArea.classList.add('dragover'), false);
                });

                ['dragleave', 'drop'].forEach(eventName => {
                    dropArea.addEventListener(eventName, () => dropArea.classList.remove('dragover'), false);
                });

                dropArea.addEventListener('drop', (e) => {
                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        input.files = files;
                        this.showNotification(`File "${files[0].name}" selected successfully!`, 'success');
                    }
                });
            }
        });
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Risk Level Animation
    animateRiskLevel(element, riskValue, duration = 1000) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }

        const startValue = 0;
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const currentValue = startValue + (riskValue - startValue) * this.easeOutCubic(progress);
            
            element.textContent = Math.round(currentValue) + '%';
            
            // Update color based on risk level
            if (currentValue < 25) {
                element.className = element.className.replace(/risk-(low|medium|high)/, 'risk-low');
            } else if (currentValue < 50) {
                element.className = element.className.replace(/risk-(low|medium|high)/, 'risk-medium');
            } else {
                element.className = element.className.replace(/risk-(low|medium|high)/, 'risk-high');
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    // Pulse Animation for Important Elements
    pulseElement(element, duration = 2000) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }

        element.classList.add('animate-pulse');
        setTimeout(() => {
            element.classList.remove('animate-pulse');
        }, duration);
    }

    // Heartbeat Animation for Health-related Elements
    heartbeatElement(element, duration = 3000) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }

        element.classList.add('animate-heartbeat');
        setTimeout(() => {
            element.classList.remove('animate-heartbeat');
        }, duration);
    }

    // Stagger Animation for Lists
    staggerAnimation(elements, delay = 100) {
        if (typeof elements === 'string') {
            elements = document.querySelectorAll(elements);
        }

        elements.forEach((element, index) => {
            setTimeout(() => {
                element.classList.add('animate-fadeInUp');
            }, index * delay);
        });
    }

    // Chart Animation Helper
    animateChart(chartElement, delay = 500) {
        if (typeof chartElement === 'string') {
            chartElement = document.querySelector(chartElement);
        }

        setTimeout(() => {
            chartElement.classList.add('chart-container');
        }, delay);
    }

    // Loading Overlay
    showLoadingOverlay(message = 'Processing...') {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        overlay.innerHTML = `
            <div class="bg-white rounded-2xl p-8 flex flex-col items-center gap-4 max-w-sm mx-4">
                <div class="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
                <p class="text-gray-700 font-medium">${message}</p>
            </div>
        `;
        document.body.appendChild(overlay);
    }

    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }
}

// Initialize Animation Manager
const animationManager = new AnimationManager();

// Export for global use
window.animationManager = animationManager;

// Additional CSS for fadeOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from { opacity: 1; transform: translateX(0); }
        to { opacity: 0; transform: translateX(100%); }
    }
`;
document.head.appendChild(style);

// Auto-initialize scroll reveal elements
document.addEventListener('DOMContentLoaded', () => {
    // Add scroll-reveal class to cards and sections
    document.querySelectorAll('.bg-white\\/80, .card-hover, section').forEach(el => {
        el.classList.add('scroll-reveal');
    });
});
