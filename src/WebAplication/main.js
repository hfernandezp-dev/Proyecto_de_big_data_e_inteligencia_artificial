$(document).ready(function() {
    EjecutarFlow();
    initApp();
});

function EjecutarFlow(){
    $.ajax({
        type: "post",
        url: "http://localhost:8000/run-pipeline",
        dataType: "json",
        success: function (response) {
            console.log("Pipeline ejecutado:", response);
        }
    });
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
        if(token) {
            // Aquí podrías hacer tu AJAX para enviar el token al backend
            console.log("Token enviado:", token);
            alert("Token enviado correctamente");

            $.ajax({
                type: "get",
                url: "http://localhost:8000/callback",
                data: { code: token },
                dataType: "json",
                success: function (response) {
                    console.log("Respuesta del servidor:", response);
                    const accessToken = response.access_token;
                    $.ajax({
                        type: "get",
                        url: "http://localhost:8000/api/recent-tracks",
                        data: {access_token:accessToken},
                        dataType: "json",
                        success: function (response) {
                            console.log("Datos de canciones:", response);
                            /*const $trackList = $("#track-list");
                            $trackList.empty();
                            response.forEach(track => {
                            const trackHTML = `
                                <div class="col">
                                    <div class="card">
                                        <img src="${track.image}" class="card-img-top" alt="${track.name}">
                                        <div class="card-body">
                                            <h5 class="card-title">${track.name}</h5>
                                            <p class="card-text">${track.artists.join(", ")}</p>
                                            <p class="card-text"><small class="text-muted">${new Date(track.played_at).toLocaleString()}</small></p>
                                        </div>
                                    </div>
                                </div>
                            `;
                            $trackList.append(trackHTML);
                        });*/

                          const container = d3.select("#chart");
                            container.selectAll("*").remove(); // limpiar gráfico previo

                            // Creamos tarjetas con D3 usando el JSON de response
                            const cards = container.selectAll(".song")
                                .data(response)
                                .enter()
                                .append("div")
                                .attr("class", "song")
                                .style("margin", "10px")
                                .style("border", "1px solid #ccc")
                                .style("padding", "10px")
                                .style("display", "flex")
                                .style("align-items", "center");

                            // Imagen de la canción
                            cards.append("img")
                                .attr("src", d => d.image)
                                .attr("width", 50)
                                .attr("height", 50)
                                .style("margin-right", "10px");

                            // Información de la canción
                            cards.append("div")
                                .html(d => `
                                    <strong>${d.name}</strong><br>
                                    ${d.artists.join(", ")}<br>
                                    <small>${new Date(d.played_at).toLocaleString()}</small>
                                `);


                        }
                    });


                }
            });

        } else {
            alert("Introduce un token válido");
        }
    });
}
