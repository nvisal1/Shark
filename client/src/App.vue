<template>
  <div class='shark'>
    <div class='shark__form-container'>
      <Form/>
    </div>
    <div class='shark__top-tooth-container'>
      <Tooth backgroundColor='#1CA7C6' opacity=0.7></Tooth>
    </div>
    <div class='shark__bottom-tooth-container'>
      <Tooth backgroundColor='#2C7080'></Tooth>
    </div>
  </div>
</template>

<script>
import Form from './components/Form.vue'
import Tooth from './components/Tooth';

export default {
  name: 'app',
  components: {
    Form,
    Tooth
  }
}
</script>

<style>
  body {
      margin: 0;
  }
  .shark {
    font-family: 'Avenir', Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
    background-color: #26C0C0;
    width: 100%;
    height: 100%;
    position: absolute;
  }
  .shark__form-container {
    position: absolute;
    left: 10%;
    top: 20%;
  }
  .shark__top-tooth-container {
    position: absolute;
    right: 0;
    bottom: 50%;
    z-index: 99;
  }
  .shark__bottom-tooth-container {
    position: absolute;
    right: 0;
    top: 65%;
  }
</style>
