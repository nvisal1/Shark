<template>
    <button class='form-button'>
        {{ buttonText }}
    </button>
</template>

<script>
export default {
    name: 'FormButton',
    props: {
        buttonText: String,
    },
}
</script>

<style scoped>
    .form-button {
        background-color: #1CA7C6;
        border: none;
        border-radius: 20px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.16), 0 3px 6px rgba(0,0,0,0.23);
        color: white;
        height: 70px;
        width: 163px;
        outline: none;
        font-size: 24px;
    }
</style>