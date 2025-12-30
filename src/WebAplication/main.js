$(document).ready(function() {
  const $MensajeError = $("#mensajeError");
  $MensajeError.hide();
  

   // EjecutarFlow();
     initApp(); 
});

function EjecutarFlow(){
  const $MensajeError = $("#mensajeError");
    $.ajax({
        type: "get",
        url: "http://localhost:8000/runflow",
        dataType: "json",
        headers: {
                "x-api-key": "spotify123"
              },
        success: function (response) {
            console.log("Pipeline ejecutado:", response);
        },
        error: function(xhr, status, error) {

            console.error("Error al ejecutar el pipeline:", error);
            $MensajeError.text("Error al ejecutar el pipeline: " + error).show();
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



function  mostrarRecomensaciones(accessToken){
  const $MensajeError = $("#mensajeError");
 //se ha decidido poner la clave api que aunque sea una vulnerabilidad es la mejor forma sin hacer un cambio significativo debido a los plazos
  $.ajax({
    type: "get",
    url: "http://localhost:8000/obtener_datos_emul",
    headers: {
    "x-api-key": "spotify123"
  },
    dataType: "json",
    success: function (response) {
      console.log("Datos de obtener2_datos_emul:", response);
      //se va ha hacer el entreanemiento
      cancionEnviar={
        instrumentalness: Number(response.instrumentalness),
        speechiness: Number(response.speechiness),
        danceability: Number(response.danceability),
        valence: Number(response.valence),
        tempo: Number(response.tempo)
      };
       console.log("Datos de cancionEnviar:", cancionEnviar);
      $.ajax({
        type: "post",
        url: "http://localhost:8000/prediccion",
        contentType: "application/json",
        data: JSON.stringify(cancionEnviar),
        headers: {
              "x-api-key": "spotify123"
            },
        dataType: "json",
        success: function (response) {
          console.log("Datos de prediccion:", response);
          const cancionesIds = [...new Set(response.recomendaciones.map(t => t.track_id).filter(Boolean))];
          console.log("ids canciones:", cancionesIds);
            $.ajax({
              type: "get",
              url: "http://localhost:8000/api/conseguir_canciones",
              data: {ids:cancionesIds.join(','), access_token: accessToken},
              dataType: "json",
               headers: {
                "x-api-key": "spotify123"
              },
              success: function (response) {
                console.log("canciones con caratulas:", response);
                const tracks = response.tracks;
                const container = $("#recomendacionesContent");
                container.empty();

                tracks.forEach((track, index) => {
                  const image = track.album.images[0]?.url;
                  const name = track.name;
                  const artist = track.artists.map(a => a.name).join(", ");
                  const spotifyUrl = track.external_urls.spotify;

                  const activeClass = index === 0 ? "active" : "";

                 const item = `
                  <div class="carousel-item ${activeClass}">
                    <div class="reco-wrapper" onclick="window.open('${spotifyUrl}', '_blank')">
                      
                      <div class="reco-blur-bg" style="background-image: url('${image}');"></div>

                      <div class="reco-content">
                        <img src="${image}" class="reco-img shadow-lg" alt="${name}">
                        <div class="reco-text">
                          <h5 class="text-truncate">${name}</h5>
                          <p class="mb-0 text-truncate">${artist}</p>
                        </div>
                      </div>

                    </div>
                  </div>
                `;
                  container.append(item);
              
                });








              },
              error: function( error) {

              console.error("Error al obtener las canciones", error);
              $MensajeError.text("Error al obtener las canciones").show();
              }
            });
          
        },
         error: function( error) {

            console.error("Error al obtener la prediccion", error);
            $MensajeError.text("Error al obtener la prediccion").show();
        }
      });
    },
     error: function( error) {

            console.error("Error al obtener los datos", error);
            $MensajeError.text("Error al obtener los datos").show();
        }
  });



}



function initApp() {
    const $MensajeError = $("#mensajeError");
    const $loginBtn = $("#login-btn");
    const $accessTokenInput = $("#access-token");
    const $enviarTokenBtn = $("#enviarToken");

    const $graficosContainer = $("#graficosConteiner");


    //se oculta el input al principio
    $accessTokenInput.hide();
    $enviarTokenBtn.hide();
    $graficosContainer.hide();

    
    let isLoggedIn = false;

    if (isLoggedIn) {
        $accessTokenInput.show();
        $enviarTokenBtn.show();
    }

    
    $loginBtn.on("click", function() {
        $MensajeError.hide();
        alert("Después de hacer login en Spotify, copia el token de acceso de la URL y pégalo en el campo de texto que se mostrara");
        window.open("http://localhost:8000/login", "_blank");
        $accessTokenInput.show();
        
        isLoggedIn = true;
        $enviarTokenBtn.show();
    });

    // Evento para enviar token
   $enviarTokenBtn.on("click", function() {
    const token = $accessTokenInput.val();
    if (token) {
        $accessTokenInput.hide();
        $enviarTokenBtn.hide();
        console.log("Token enviado:", token);
        alert("Token enviado correctamente");

        $.ajax({
            type: "get",
            url: "http://localhost:8000/callback",
            data: { code: token },
            headers: {
                "x-api-key": "spotify123"
              },
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
                    headers: {
                      "x-api-key": "spotify123"
                    },
                    success: function (recentTracks) { 
                        console.log("Datos de canciones:", recentTracks);
                        $graficosContainer.show();
                        const chartData = prepareChartData(recentTracks);
                        drawBarChart(chartData,"#chart");

                        const artistIds = [...new Set(recentTracks.map(t => t.artist_id).filter(Boolean))];
                        console.log("IDs únicos de artistas:", artistIds);
                        $.ajax({
                            type: "get",
                            url: "http://localhost:8000/api/generos",
                            data: { ids: artistIds.join(','), access_token: accessToken }, 
                            dataType: "json",
                            headers: {
                              "x-api-key": "spotify123"
                            },
                            success: function (genresResponse) {
                                console.log("Datos de géneros:", genresResponse);
                                const genreData = prepareGenresData(genresResponse);
                                drawBarChart(genreData,"#genre-chart");
                                mostrarRecomensaciones(accessToken);
                            },
                            error: function( error) {
                            console.error("Error al obtener los generos", error);
                            $MensajeError.text("Error al obtener los generos").show();
                            }
                        });
                    },
                    error: function( error) {
                    console.error("Error al obtener las canciones recientes", error);
                    $MensajeError.text("Error al obtener las canciones recientes").show();
                    }
                });

            },
            error: function( error) {

              console.error("Error al obtener el accesstoken", error);
              $MensajeError.text("Error al obtener el accesstoken").show();
            }
        });

    } else {
        alert("Introduce un token válido");
    }
});
}
