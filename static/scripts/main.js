document.addEventListener("DOMContentLoaded", () => {
    // Mobile Menu Toggle
    const mobileMenuBtn = document.getElementById("mobileMenuBtn")
    const mobileNav = document.getElementById("mobileNav")
  
    mobileMenuBtn.addEventListener("click", () => {
      mobileNav.classList.toggle("active")
  
      // Toggle icon
      const icon = mobileMenuBtn.querySelector("i")
      if (icon.classList.contains("fa-bars")) {
        icon.classList.remove("fa-bars")
        icon.classList.add("fa-times")
      } else {
        icon.classList.remove("fa-times")
        icon.classList.add("fa-bars")
      }
    })
  
    // Chat Bot Toggle
    const chatToggle = document.getElementById("chatToggle")
    const chatWindow = document.getElementById("chatWindow")
    const chatIcon = document.getElementById("chatIcon")
    const closeIcon = document.getElementById("closeIcon")
  
    chatToggle.addEventListener("click", () => {
      const isOpen = chatWindow.style.display === "flex"
  
      if (isOpen) {
        chatWindow.style.display = "none"
        chatIcon.style.display = "block"
        closeIcon.style.display = "none"
      } else {
        chatWindow.style.display = "flex"
        chatIcon.style.display = "none"
        closeIcon.style.display = "block"
      }
    })
  
    // Chat Form Submission
    const chatForm = document.getElementById("chatForm")
    const chatInput = document.getElementById("chatInput")
    const chatMessages = document.getElementById("chatMessages")
  
    chatForm.addEventListener("submit",async (e) => {
      e.preventDefault()
  
      const message = chatInput.value.trim()
      if (!message) return
  
      // Add user message
      addMessage(message, "user")
      chatInput.value = ""
      try {
          // Send message to Flask backend
          const response = await fetch("http://localhost:5000/chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ message }),
          });

          const data = await response.json();

          // Add bot response to chat window
          if (data.response) {
              addMessage(data.response, "bot");
          } else {
              addMessage("Sorry, I couldn't process that request.", "bot");
          }
      } catch (error) {
          console.error("Error communicating with the chatbot:", error);
          addMessage("Error connecting to chatbot.", "bot");
      }
    });
  
    function addMessage(text, sender) {
      const messageDiv = document.createElement("div")
      messageDiv.classList.add("message", `${sender}-message`)
  
      const messageContent = document.createElement("div")
      messageContent.classList.add("message-content")
      if (sender === "bot") {
          messageContent.textContent = "Typing...";
          messageDiv.appendChild(messageContent);
          chatMessages.appendChild(messageDiv);
          chatMessages.scrollTop = chatMessages.scrollHeight;

          // Simulate a delay before displaying the full response
          setTimeout(() => {
              messageContent.textContent = "";
              typeMessage(text, messageContent);
          }, 1000); // 1s delay for typing effect
      } else {
          messageContent.textContent = text;
          messageDiv.appendChild(messageContent);
          chatMessages.appendChild(messageDiv);
      }

      // Scroll to bottom
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function typeMessage(text, element, index = 0) {
        if (index < text.length) {
            element.textContent += text[index];
            setTimeout(() => typeMessage(text, element, index + 1), 30); // Adjust speed here
        }
    }
  
    // Animation for step cards
    const stepCards = document.querySelectorAll(".step-card")
  
    function checkScroll() {
      stepCards.forEach((card) => {
        const cardTop = card.getBoundingClientRect().top
        const windowHeight = window.innerHeight
  
        if (cardTop < windowHeight * 0.8) {
          card.style.opacity = "1"
          card.style.transform = "translateY(0)"
        }
      })
    }
  
    // Initialize step cards
    stepCards.forEach((card) => {
      card.style.opacity = "0"
      card.style.transform = "translateY(20px)"
      card.style.transition = "opacity 0.5s ease, transform 0.5s ease"
    })
  
    // Check on load and scroll
    checkScroll()
    window.addEventListener("scroll", checkScroll)
  })
  
  