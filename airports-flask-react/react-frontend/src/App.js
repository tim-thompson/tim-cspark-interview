import './App.css';
import { useEffect, useState } from 'react';
import GoogleMapReact from 'google-map-react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faLocationDot, faPlane } from '@fortawesome/free-solid-svg-icons'

function App() {
  const [latitude, setLatitude] = useState(0.0);
  const [longitude, setLongitude] = useState(0.0);
  const [data, setData] = useState();

  useEffect(() => {
    fetch(`http://127.0.0.1:8000/nearest_airport?latitude=${latitude}&longitude=${longitude}`)
      .then(response => response.json())
      .then(data => setData(data));
  }, [latitude, longitude])

  const handleMapClick = async (event) => {
    setLatitude(event.lat);
    setLongitude(event.lng);
  };

  const defaultProps = {
    center: {
      lat: 51.569065,
      lng: 0.407348
    },
    zoom: 7
  };

  const Marker = ({text}) => (
    <FontAwesomeIcon icon={faLocationDot} size={'xl'}/>
  );

  const AirportMarker = ({text}) => (
    <FontAwesomeIcon icon={faPlane} size={'xl'}/>
  );

  return (
    <div className="App" style={{ height: '70vh', width: '100%' }}>
      <GoogleMapReact
        bootstrapURLKeys={{ key: "REDACTED" }}
        defaultCenter={defaultProps.center}
        defaultZoom={defaultProps.zoom}
        onClick={handleMapClick}
      >
        <Marker
          text={"Test"}
          lat={latitude}
          lng={longitude}
        />
        {data && (
          <AirportMarker
            text={"Airport"}
            lat={data.airport.latitude}
            lng={data.airport.longitude}
          />
        )}
        
      </GoogleMapReact>
      <form>
        <p>
          Click the map to set a location. 
          <br></br>
          The nearest airport will be displayed on the map and 
          the details will be displayed below.
          <br></br>
          Alternatively provide specific coordinates using the boxes.
          <br></br>
          The map will display updates automatically. 
        </p>
        <label>Latitude:
          <input 
            type="text" 
            value={latitude}
            onChange={(e) => setLatitude(e.target.value)}
          />
        </label>
        <label>Longitude:
          <input 
            type="text" 
            value={longitude}
            onChange={(e) => setLongitude(e.target.value)}
          />
        </label>
      </form>
      {data && (
        <div>
          <br></br><b>Airport: </b> {data.airport.name}
          <br></br><b>ICAO: </b> {data.airport.ICAO}
          <br></br><b>Distance (KM): </b> {Math.round(data.distance_km)}
        </div>
      )}
    </div>
  );
}

export default App;
