<template>
  <div class='form'>
    <div class='form__title-container'>
      <FormTitle message='Shark'></FormTitle>
    </div>
    <div class='form__inputs-container'>
      <div class='form__inputs-container__input-container' v-for='data in inputData' :key='data'>
        <FormInput 
          :placeholderText='data.placeholderText'
        ></FormInput>
      </div>
    </div>
    <FormButton buttonText='Predict'></FormButton>
  </div>
</template>

<script>
  import FormButton from './Form-Button';
  import FormInput from './Form-Input';
  import FormTitle from './Form-Title';

  const inputData = [
    {
      placeholderText: 'R&D Spend'
    },
    {
      placeholderText: 'Administration Spend'
    },
    {
      placeholderText: 'Marketing Spend'
    },
    {
      placeholderText: 'State'
    },
  ];

  export default {
    name: 'Form',
    components: {
      FormButton,
      FormInput,
      FormTitle,
    },
    data: function() {
      return { inputData }
    },
  }
</script>

<style scoped>
  .form__title-container {
    margin-bottom: 55px;
  }
  .form__inputs-container {
    margin-bottom: 55px;
  }
  .form__inputs-container__input-container{
    margin-bottom: 31px;
  }
</style>
