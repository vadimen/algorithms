import React, {Component} from 'react';
import {Platform, StyleSheet, Text, View} from 'react-native';
import MapView, {PROVIDER_GOOGLE} from 'react-native-maps';
import Marker from 'react-native-maps';

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'flex-end',
    alignItems: 'center',
  },
  map: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
});

class Human{
  static next_id: number = 0;
  id: number;
  name: string;
  lat: number;
  lon: number;
  alive: boolean;

  constructor(lt: number,
              ln: number,
              nm: string){
    this.lat = lt;
    this.lon = ln;
    this.name = nm;
    alive = true;
    this.id = Human.next_id;
    Human.next_id = Human.next_id + 1;
  }

  kill(){
    this.lat = 0;
    this.lon = 0;
    this.alive = false;
  }

  clock(){
    if(this.alive == false) return;

    var new_lat:number;
    var new_lon:number;
    var speed:number = 0.00001;

    //check dist between this human and terminator
    //if closer than 10 steps then it's followed
    //and must run
    var dist:number = Math.sqrt(
      Math.pow(this.lat-term.lat,2)+Math.pow(this.lon-term.lon,2)
      )

    var followed:boolean = false;
    if(dist < this.lat*(speed*10)) //less than ten human steps(10%)
      followed=true;

    if(followed == false){
      var rand_nr:number = Math.floor(Math.random()*100)+1;

      //four margin points
      // 37.78825,-122.4724
      // 37.78825,-122.4024
      // 37.74825,-122.4024
      // 37.74825,-122.4724

      if(rand_nr >= 0 && rand_nr < 25){
        new_lat = this.lat + this.lat*speed;
        new_lon = this.lon + this.lon*speed;
      }else if(rand_nr >= 25 && rand_nr < 50){
        new_lat = this.lat + this.lat*speed;
        new_lon = this.lon - this.lon*speed;
      }else if(rand_nr >= 50 && rand_nr < 75){
        new_lat = this.lat - this.lat*speed;
        new_lon = this.lon + this.lon*speed;
      }else if(rand_nr >= 75){
        new_lat = this.lat - this.lat*speed;
        new_lon = this.lon - this.lon*speed;
      }
    }else{//run from terminator
      if(this.lat > term.lat)
        new_lat = this.lat + this.lat*speed*2;
      else if(this.lat < term.lat)
        new_lat = this.lat - this.lat*speed*2;
      if(this.lon > term.lon)
        new_lon = this.lon + this.lon*speed*2;
      else if(this.lon < term.lon)
        new_lon = this.lon - this.lon*speed*2;
    }

    if(new_lat >= 37.74825 && new_lat <= 37.78825)
      this.lat = new_lat;
    if(new_lon >= -122.4724 && new_lon <= -122.4024)
      this.lon = new_lon;
  }
};

let humans : Array<Human> = [];
humans.push(new Human(37.76825,-122.4374, "Ivan"));
humans.push(new Human(37.76825,-122.4374, "Vanea"));
humans.push(new Human(37.76825,-122.4374, "Vano"));
humans.push(new Human(37.76825,-122.4374, "Vieno"));
humans.push(new Human(37.76825, -122.4374, "Ghenadie"));
humans.push(new Human(37.76825, -122.4374, "Grigorie"));
humans.push(new Human(37.76825, -122.4374, "Zinaida"));


class Terminator{
  name: string;
  lat: number;
  lon: number;

  constructor(lt: number,
              ln: number,
              nm: string){
    this.lat = lt;
    this.lon = ln;
    this.name = nm;
  }

  clock(){
    
    var dist:number;

    var new_lat:number;
    var new_lon:number;
    var speed:number = 0.00002;

    //calc dist to every human
    for(let hm of humans){
      dist = Math.sqrt(
        Math.pow(this.lat-hm.lat,2)+Math.pow(this.lon-hm.lon,2)
      )
      if(dist < this.lat*(speed*5)){ //less than 5 terminator steps
        humans[hm.id].kill();
      }
    }

    
    var rand_nr:number = Math.floor(Math.random()*100)+1;

    if(rand_nr >= 0 && rand_nr < 25){
      new_lat = this.lat + this.lat*speed;
      new_lon = this.lon + this.lon*speed;
    }else if(rand_nr >= 25 && rand_nr < 50){
      new_lat = this.lat + this.lat*speed;
      new_lon = this.lon - this.lon*speed;
    }else if(rand_nr >= 50 && rand_nr < 75){
      new_lat = this.lat - this.lat*speed;
      new_lon = this.lon + this.lon*speed;
    }else if(rand_nr >= 75){
      new_lat = this.lat - this.lat*speed;
      new_lon = this.lon - this.lon*speed;
    }

    if(new_lat >= 37.74825 && new_lat <= 37.78825)
      this.lat = new_lat;
    if(new_lon >= -122.4724 && new_lon <= -122.4024)
      this.lon = new_lon;
  }
}

let term:Terminator = new Terminator(37.78825, -122.4724, "Schwarz");

class App extends React.Component {

  constructor(props){
    super(props);
    this.state = {
      updated: 0
    };
  }

  componentDidMount(){
    this.timeoutHandle = setInterval(
      () => {
        for(let hm of humans){
          hm.clock();
        }
        term.clock();
        this.setState({state: this.state})
      },50
    );
  }

  mapHumans = () => {
    return humans.map((human) => 
      <MapView.Marker
        key={human.id}
        title={human.name}
        subtitle={human.name}
        pinColor = {"purple"}
        description={human.name}
        coordinate={{
          latitude: human.lat,
          longitude: human.lon}}
      />
    )
  }

  render() {
    return (
      <MapView
        style={ styles.map }
        initialRegion={{
          latitude: 37.78825,
          longitude: -122.4324,
          latitudeDelta: 0.0922,
          longitudeDelta: 0.0421,
        }}
      >
      {this.mapHumans()}
      <MapView.Marker
        title={term.name}
        subtitle={term.name}
        pinColor = {"red"}
        description={term.name}
        coordinate={{
          latitude: term.lat,
          longitude: term.lon}}
      />
      </MapView>
    );
  }
}


export default App;