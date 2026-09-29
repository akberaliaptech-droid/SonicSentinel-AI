// Mobile Menu Toggle
function toggleMenu() {
    const nav = document.getElementById('navLinks');
    if (window.innerWidth <= 768) {
        nav.classList.toggle('active');
    }
}

// Modal Logic
function openLoginModal() {
    document.getElementById('loginModal').classList.add('active');
}

function closeLoginModal() {
    document.getElementById('loginModal').classList.remove('active');
}

// Handle Login and View Switching
function handleLogin(event) {
    event.preventDefault();
    
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();
    
    let role = null;
    
    if (username === 'admin' && password === 'admin123') {
        role = 'admin';
    } else if (username === 'user' && password === 'user123') {
        role = 'user';
    } else {
        alert("Invalid Username or Password! Please use the hint.");
        return;
    }
    
    // Hide Landing View
    document.getElementById('landingView').style.display = 'none';
    
    // Update Navbar Buttons
    document.getElementById('loginNavBtn').style.display = 'none';
    document.getElementById('logoutNavBtn').style.display = 'block';
    
    // Redirect to the actual Dashboard page
    if (role === 'admin') {
        alert("Welcome Admin! Redirecting to Dashboard...");
        window.location.href = "/dashboard";
    } else {
        alert("Welcome User! Redirecting to Dashboard...");
        window.location.href = "/dashboard";
    }
}

// Handle Logout
function handleLogout() {
    // Reset Views
    document.getElementById('landingView').style.display = 'block';
    document.getElementById('adminView').style.display = 'none';
    document.getElementById('userView').style.display = 'none';
    
    // Update Navbar Buttons
    document.getElementById('loginNavBtn').style.display = 'block';
    document.getElementById('logoutNavBtn').style.display = 'none';
    
    window.scrollTo(0, 0);
}

// Role Edit Functionality for Admin
function editRole(button) {
    const row = button.parentElement.parentElement;
    const roleBadge = row.querySelector('.role-badge');
    const currentRole = roleBadge.innerText;
    
    if (currentRole === 'User') {
        roleBadge.className = 'role-badge role-admin';
        roleBadge.innerText = 'Admin';
        button.innerText = 'Make User';
    } else {
        roleBadge.className = 'role-badge role-user';
        roleBadge.innerText = 'User';
        button.innerText = 'Make Admin';
    }
    
    // Trigger small animation
    roleBadge.style.transform = 'scale(1.2)';
    setTimeout(() => {
        roleBadge.style.transform = 'scale(1)';
    }, 200);
}

// Close modal if clicked outside
window.onclick = function(event) {
    const modal = document.getElementById('loginModal');
    if (event.target == modal) {
        closeLoginModal();
    }
}
