$(document).ready(function() {
   // EjecutarFlow();
    initApp();
});

function EjecutarFlow(){
    $.ajax({
        type: "get",
        url: "http://localhost:8000/runflow",
        dataType: "json",
        success: function (response) {
            console.log("Pipeline ejecutado:", response);
        }
    });
}

function drawGenrePieChart(data) {
  const container = document.querySelector('#genre-chart'); 
  const width = container.clientWidth;
  const height = 400;

  d3.select('#genre-chart').selectAll('*').remove();

  const svg = d3.select('#genre-chart')
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .style('max-width', '100%')
    .style('height', 'auto')
    .style('overflow', 'visible')
    .append('g')
    .attr('transform', `translate(${width / 2}, ${height / 2})`);

  const radius = Math.min(width, height) / 2 * 0.8;
  const color = d3.scaleOrdinal(d3.schemeCategory10);
  const pie = d3.pie().sort(null).value(d => d.value);
  const arc = d3.arc().innerRadius(0).outerRadius(radius);
  const outerArc = d3.arc().innerRadius(radius * 1.05).outerRadius(radius * 1.05);

  const pieData = pie(data.slice(0, 10)); // TOP 10 géneros

  // Sectores
  svg.selectAll('path')
    .data(pieData)
    .enter()
    .append('path')
    .attr('d', arc)
    .attr('fill', d => color(d.data.name))
    .attr('stroke', '#fff')
    .style('stroke-width', '2px');

  // Líneas guía
  svg.selectAll('polyline')
    .data(pieData)
    .enter()
    .append('polyline')
    .attr('points', d => {
      const posA = arc.centroid(d);
      const posB = outerArc.centroid(d);
      const posC = [...posB];
      posC[0] = radius * 1.2 * (d.startAngle + d.endAngle < Math.PI ? 1 : -1);
      return [posA, posB, posC];
    })
    .style('fill', 'none')
    .style('stroke', '#999')
    .style('stroke-width', '1px');

  // Etiquetas externas con ajuste vertical
  const texts = svg.selectAll('text')
    .data(pieData)
    .enter()
    .append('text')
    .text(d => d.data.name)
    .attr('font-size', '11px')
    .attr('text-anchor', d => (d.startAngle + d.endAngle < Math.PI ? 'start' : 'end'))
    .each(function(d, i) {
      const pos = outerArc.centroid(d);
      pos[0] = radius * 1.2 * (d.startAngle + d.endAngle < Math.PI ? 1 : -1);
      d.yPos = pos[1];
      d3.select(this).attr('transform', `translate(${pos})`);
    });

  // Ajustar solapamiento vertical
  const spacing = 14; // px entre etiquetas
  const left = texts.filter(d => d.startAngle + d.endAngle >= Math.PI).nodes();
  const right = texts.filter(d => d.startAngle + d.endAngle < Math.PI).nodes();

  [left, right].forEach(nodes => {
    nodes.sort((a, b) => d3.select(a).attr('transform').split(',')[1] - d3.select(b).attr('transform').split(',')[1]);
    for(let i = 1; i < nodes.length; i++){
      const prev = nodes[i-1];
      const curr = nodes[i];
      const prevY = parseFloat(prev.getAttribute('transform').split(',')[1]);
      const currY = parseFloat(curr.getAttribute('transform').split(',')[1]);
      if(currY - prevY < spacing){
        const delta = spacing - (currY - prevY);
        const t = curr.getAttribute('transform').replace(/translate\(([^)]+)\)/, (m, g) => {
          const [x, y] = g.split(',').map(Number);
          return `translate(${x},${y + delta})`;
        });
        curr.setAttribute('transform', t);
      }
    }
  });
}



function drawBarChart(data,selector) {
   const container = document.querySelector(selector);
  const width = container.clientWidth; // Ancho del div
  const height = 400;
  const margin = { top: 20, right: 20, bottom: 100, left: 60 };

  // Limpiar contenido previo
  d3.select(selector).selectAll('*').remove();

  // Crear SVG responsive
  const svg = d3.select(selector)
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .style('max-width', '100%')
    .style('height', 'auto')
    .style('overflow', 'visible')
    .append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

  const chartWidth = width - margin.left - margin.right;
  const chartHeight = height - margin.top - margin.bottom;

  // Escalas
  const x = d3.scaleBand()
    .domain(data.map(d => d.name))
    .range([0, chartWidth])
    .padding(0.2);

  const y = d3.scaleLinear()
    .domain([0, d3.max(data, d => d.value)])
    .nice()
    .range([chartHeight, 0]);

  // Barras
  svg.selectAll('rect')
    .data(data)
    .enter()
    .append('rect')
    .attr('x', d => x(d.name))
    .attr('y', d => y(d.value))
    .attr('width', x.bandwidth())
    .attr('height', d => chartHeight - y(d.value))
    .attr('fill', '#1db954');

  // Eje X
  svg.append('g')
    .attr('transform', `translate(0,${chartHeight})`)
    .call(d3.axisBottom(x))
    .selectAll('text')
    .attr('transform', 'rotate(-40)')
    .style('text-anchor', 'end');

  // Eje Y SIN decimales
  svg.append('g')
    .call(
      d3.axisLeft(y)
        .ticks(d3.max(data, d => d.value))
        .tickFormat(d3.format('d')) // ❌ clave: quitar decimales
    );
}


function prepareGenresData(genresResponse) {
  const counter = {};

  genresResponse.artists.forEach(artist => {
    artist.genres.forEach(genre => {
      counter[genre] = (counter[genre] || 0) + 1;
    });
  });

  return Object.entries(counter).map(([name, value]) => ({
    name,
    value
  }));
}

function prepareChartData(tracks) {
  const counter = {};

  tracks.forEach(track => {
    const artist = track.artists[0] || 'Desconocido';
    counter[artist] = (counter[artist] || 0) + 1;
  });

  return Object.entries(counter).map(([name, value]) => ({
    name,
    value
  }));
}

function prepareSongsChartData(tracks) {
  const counter = {};

  genresResponse.artists.forEach(artist => {
    artist.genres.forEach(genre => {
      counter[genre] = (counter[genre] || 0) + 1;
    });
  });

  // Convertir a array y ordenar de mayor a menor
  const sortedGenres = Object.entries(counter)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  // Devolver solo los top N
  return sortedGenres.slice(0, topN);
}





function initApp() {
    const $loginBtn = $("#login-btn");
    const $accessTokenInput = $("#access-token");
    const $enviarTokenBtn = $("#enviarToken");

    // Inicialmente ocultamos input y botón de enviar token
    $accessTokenInput.hide();
    $enviarTokenBtn.hide();

    // Variable simulando estado de login
    let isLoggedIn = false;

    // Si el usuario ya está logueado, mostramos el token y botón
    if (isLoggedIn) {
        $accessTokenInput.show();
        $enviarTokenBtn.show();
    }

    // Evento para botón de login
    $loginBtn.on("click", function() {
        window.open("http://localhost:8000/login", "_blank");
        alert("Después de hacer login en Spotify, vuelve aquí y presiona OK para cargar tus canciones.");

        // Simulamos login
        isLoggedIn = true;
        $accessTokenInput.val("TOKEN_SIMULADO").show();
        $enviarTokenBtn.show();
    });

    // Evento para enviar token
   $enviarTokenBtn.on("click", function() {
    const token = $accessTokenInput.val();
    if (token) {
        console.log("Token enviado:", token);
        alert("Token enviado correctamente");

        // PASO 1: Intercambio de Código (Callback)
        $.ajax({
            type: "get",
            url: "http://localhost:8000/callback",
            data: { code: token },
            dataType: "json",
            success: function (authResponse) { // <-- RENOMBRAR: authResponse
                console.log("Respuesta del servidor (Token):", authResponse);
                const accessToken = authResponse.access_token; // <-- Usamos authResponse
                
                // PASO 2: Obtener Canciones Recientes
                $.ajax({
                    type: "get",
                    url: "http://localhost:8000/api/recent-tracks",
                    data: { access_token: accessToken },
                    dataType: "json",
                    success: function (recentTracks) { 
                        console.log("Datos de canciones:", recentTracks);
                        const chartData = prepareChartData(recentTracks);
                        drawBarChart(chartData,"#chart");

                        const artistIds = [...new Set(recentTracks.map(t => t.artist_id).filter(Boolean))];
                        console.log("IDs únicos de artistas:", artistIds);
                        $.ajax({
                            type: "get",
                            url: "http://localhost:8000/api/generos",
                            data: { ids: artistIds.join(','), access_token: accessToken }, 
                            dataType: "json",
                            success: function (genresResponse) {
                                console.log("Datos de géneros:", genresResponse);
                                const genreData = prepareGenresData(genresResponse);
                                drawBarChart(genreData,"#genre-chart");
                            }
                        });

                     /*    // --- A. DIBUJAR TARJETAS (Usando el ARRAY CORRECTO) ---
                        const container = d3.select("#chart");
                        container.selectAll("*").remove(); 
                        
                        // NOTA: Usar 'recentTracks' en lugar de 'response'
                        const cards = container.selectAll(".song")
                            .data(recentTracks) 
                            .enter()
                            // ... (resto de tu código D3 para tarjetas) ...
                            .append("div")
                            .attr("class", "song")
                            .style("margin", "10px")
                            // ... (estilos)
                            .html(d => `
                                <strong>${d.name}</strong><br>
                                ${d.artists.join(", ")}<br>
                                <small>${new Date(d.played_at).toLocaleString()}</small>
                            `);
                        
                        // --- B. PROCESAMIENTO DE GÉNEROS (DENTRO DEL ÁMBITO CORRECTO) ---
                        const uniqueArtistIds = Array.from(new Set(
                            recentTracks // <-- ¡USAMOS EL ARRAY DE CANCIONES AQUÍ!
                                .map(track => track.artist_id)
                                .filter(id => id) 
                        ));
                        
                        if (uniqueArtistIds.length === 0) {
                            console.log("No hay artistas para consultar géneros.");
                            return;
                        } */

                      /*   $.ajax({
                            type: "get",
                            url: "http://localhost:8000/api/generos",
                            // NOTA: Recuerda que `ids` debe ser una cadena separada por comas en la URL
                            data: { ids: uniqueArtistIds.join(','), access_token: accessToken }, 
                            dataType: "json",
                            success: function (genresResponse) {
                                console.log("Datos de géneros:", genresResponse);
                                // Aquí puedes llamar a tu función processDataForD3(recentTracks, genresResponse)
                            }
                        }); */
                    }
                });

            }
        });

    } else {
        alert("Introduce un token válido");
    }
});
}
