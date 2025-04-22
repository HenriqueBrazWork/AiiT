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
