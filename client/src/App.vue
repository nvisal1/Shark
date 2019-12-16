<template>
  <div class="shark">
    <div class='shark__form-container'>
      <Form/>
    </div>
  </div>
</template>

<script>
import Form from './components/Form.vue'

export default {
  name: 'app',
  components: {
    Form
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
</style>
