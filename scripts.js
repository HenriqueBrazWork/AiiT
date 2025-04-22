// Toggle menu mobile ao clicar no logo
const logo = document.querySelector('#header h1');
const navList = document.querySelector('#header nav ul');
logo.addEventListener('click', () => {
  navList.classList.toggle('show');
});

// (Opcional) Auto-scroll do carrossel de depoimentos
// setInterval(() => {
//   const carousel = document.querySelector('.testimonial-carousel');
//   carousel.scrollBy({ left: carousel.offsetWidth, behavior: 'smooth' });
// }, 5000);

// Carregar a app Streamlit só ao clicar
const loadBtn = document.getElementById('load-app-btn');
loadBtn.addEventListener('click', () => {
  const iframe = document.getElementById('ai-app-frame');
  // Substitui pelo URL público da tua app no Streamlit Cloud
  iframe.src = 'https://dhryh3aww7wwjgs8poyf74.streamlit.app/';
  // Mostra o container e esconde o botão
  document.getElementById('iframe-container').style.display = 'block';
  loadBtn.style.display = 'none';
});
